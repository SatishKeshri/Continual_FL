from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import math
from torchvision import transforms
from torch.nn.utils import clip_grad_norm_
from dirichlet_sampler import sampler
from utils import arg_parser, average_weights
import gc
import random
from types import SimpleNamespace
from collections import defaultdict
import time
import timeit
from collections import Counter



import pickle
import os
import shutil

from avalanche.benchmarks.classic import SplitMNIST, PermutedMNIST, SplitCIFAR100, SplitCIFAR10, SplitTinyImageNet
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.storage_policy import ParametricBuffer, RandomExemplarsSelectionStrategy
from avalanche.benchmarks.utils.data_loader import GroupBalancedDataLoader

from models.resnet_new import ResNet18, ResNet50


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_classes: int):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MLP_multi(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_classes: int, num_tasks=5, multi_head=False):
        super(MLP_multi, self).__init__()
        # number of classes should be divisible by number of tasks
        assert n_classes % num_tasks == 0, "Number of classes should be divisible by number of tasks"

        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, n_classes)
        # self.fc2_list = 
        if multi_head:
            self.task_classifiers = nn.ModuleList([
            nn.Linear(hidden_size, n_classes//num_tasks) for _ in range(num_tasks)
            ])
        else:
            self.task_classifiers = nn.ModuleList([
            nn.Linear(hidden_size, n_classes)
            ])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, task_id, memory_flag=False) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.task_classifiers[task_id](x)
        return x

def format_dict_with_precision(data, precision=4):
    return {key: tuple(f'{val:.{precision}f}' for val in value) for key, value in data.items()}


class CFLAG:
    """Proposed algorithm with IAG"""

    def __init__(self, args: Dict[str, Any], benchmark, num_tasks, multi_head=False):
        self.args = args
        self.bm = benchmark
        self.num_tasks = num_tasks
        if self.args.dataset == "CIFAR100":
            self.num_classes_total = 100
        elif self.args.dataset == "tinyimagenet":
            self.num_classes_total = 200
        else:
            self.num_classes_total = args.num_classes_total
        # print(f"cifar in dataset name: {'cifar' in self.args.dataset.lower()}")
        # print(f"num classes total: {self.num_classes_total}")
        self.n_client_epochs = args.n_client_epochs
        self.multi_head_flag = args.multi_head_flag
        if self.multi_head_flag:
            assert self.num_classes_total % self.num_tasks == 0, "For multi-head model number of classes should be divisible by number of tasks"
            self.class_per_head = self.num_classes_total // self.num_tasks
        else:
            self.class_per_head = self.num_classes_total
        # self.multi_head_flag = args.multi_head
        self.L = args.L
        self.eps = args.eps
        self.optim_name = str(args.optim).lower()
        # self.num_classes_total = args.num_classes_total
        # self.class_per_head = args.class_per_head
        self.memory_sample_size = args.memory_sample_size
        self.clip_grad_g_val = args.clip_grad_g_val
        self.clip_grad_f_val = args.clip_grad_f_val
        self.max_clip = args.max_clip
        self.min_clip = args.min_clip
        self.clip_grad_norms = args.clip_grad_norms
        self.norm_threshold = args.norm_threshold
        self.IAG_flag  = args.IAG_flag
        self.grad_f_flag = args.grad_f_flag
        self.IAG_batch_flag = args.IAG_batch_flag
        self.delayed_grad_batch_flag = args.delayed_grad_batch_flag
        self.kick_client = args.kick_client
        self.kick_memory = args.kick_memory
        self.proxy_adaptive_lr = args.proxy_adaptive_lr
        if self.IAG_batch_flag == True:
            self.train_shuffle = False
        else:
            self.train_shuffle = True
        self.adaptive_flag = args.adaptive_flag
        self.adaptive_lr_batch_flag = args.adaptive_lr_batch_flag 
        self.adaptive_lr_step_flag = args.adaptive_lr_step_flag 
        self.adaptive_lr_round_flag = args.adaptive_lr_round_flag
        assert self.IAG_batch_flag != self.train_shuffle, "IAG_batch_flag and train_shuffle should be of opposite bool values"
        if self.IAG_batch_flag:
            assert self.delayed_grad_batch_flag, "For IAG_batch, delayed_grad_batch_flag should be True"
        if not self.grad_f_flag:
            assert not self.adaptive_flag, "For adaptive learning((adaptive_flag=True)), grad_f_flag should be True"
        if self.adaptive_lr_step_flag and self.adaptive_lr_round_flag:
            raise ValueError("Both the step and round adaptive learning flags can not be True")
        if not self.adaptive_flag:
            assert not self.proxy_adaptive_lr, "For no adaptive learning (adaptive_flag=False), proxy_adaptive_lr should be False"
        self.device = torch.device(
            f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
        # self.device = torch.device(
        #     "cpu"
        )
    
    def _calc_gamma(self, beta, ai, alp, inn, ):
        """Calculate forgetting gamma for each client"""
        ai_2 = self._inner_prod_sum(ai, ai)
        first_term = (self.L*beta*ai_2) / (2*self.num_clients)
        second_term = - beta* (1- self.L*alp) * inn

        return first_term + second_term
    
    
    
    def server(self, ):
        """Implementation of the server strategy"""
        if self.args.model_name == "MLP":
            self.global_model = MLP_multi(input_size=28*28, hidden_size=512, n_classes=10, num_tasks=self.num_tasks,multi_head=self.multi_head_flag).to(self.device)
        elif self.args.model_name == "resnet18":
            self.global_model = ResNet18(num_classes_total=self.num_classes_total, num_tasks=self.num_tasks, multi_head=self.multi_head_flag,args={"dataset":self.args.dataset}).to(self.device)
        elif self.args.model_name == "resnet50":
            self.global_model = ResNet50(num_classes_total=self.num_classes_total, num_tasks=self.num_tasks, multi_head=self.multi_head_flag,args={"dataset":self.args.dataset}).to(self.device)
        else:
            raise NotImplementedError
        print(f"Training using model: {type(self.global_model)}")
        
        self.num_clients = max(int(self.args.frac * self.args.n_clients),1)
        self.client_buffer_size = [self.args.initial_buffer_size]*self.num_clients
        self.memory_buffer = [ParametricBuffer(max_size=self.client_buffer_size[0],groupby='task',selection_strategy=RandomExemplarsSelectionStrategy())
                                                    for _ in range(self.num_clients)
                                                    ]
        self.task_store = []
        
        self.cl_models_list = [copy.deepcopy(self.global_model) for _ in range(self.num_clients)]
        self.global_gamma_dict = {}

        for task_id in range(self.num_tasks):
            self.task_id = task_id
            
            
            exp_train = self.bm.train_stream[self.task_id]
            _ = self.mini_server(exp_train)
            # calc grad_f_norm
            grads_f_dict_list = [ self._get_memory_grads_new(client_idx,grads_g=None)[0] for client_idx in range(self.num_clients)]
            grads_f_new_first_list = [ [param for name, param in grads_f_dict.items()] for grads_f_dict in grads_f_dict_list ]
            print(f"End of task: {self.task_id}, server, global model param_sum: {sum(param.sum() for param in self.global_model.parameters())}")
            if self.task_id >0:
                self.global_gamma_dict[self.task_id] = self.client_gamma
        
        # print("Training completed")
        print(f"Task-wise test metric progression dict:")
        for entry in self.task_store:
            print(entry)
        
        
        print(f"full task gamma dict:\n {self.global_gamma_dict}")
        with open(global_result_path+'task_gamma_dict.pkl', 'wb') as f:
            pickle.dump(self.global_gamma_dict, f)
        with open(global_result_path+"final_result"+'.pkl', 'wb') as f:
            pickle.dump(self.task_store, f)
        

    def create_optimizer(self, task_id, model):

        if self.optim_name.lower() == "sgd":
            if (task_id == 0):
                current_lr = self.args.lr 
                optim = torch.optim.SGD(model.parameters(),
                                            lr=self.args.lr,
                                            momentum=self.args.momentum,
                                            weight_decay=5e-4,
                                            )
                return optim
            else :
                current_lr = self.args.lr_retrain 
                optim = torch.optim.SGD(model.parameters(),
                                            lr=self.args.lr_retrain,
                                            momentum=self.args.momentum,
                                            weight_decay=5e-4,
                                            )
                return optim
        elif self.optim_name.lower() == "adam":
            if (self.task_id == 0):
                current_lr = self.args.lr 
                optim = torch.optim.Adam(model.parameters(),
                                            lr=self.args.lr,
                                            )
                return optim
            else :
                current_lr = self.args.lr_retrain 
                optim = torch.optim.Adam(model.parameters(),
                                            lr=self.args.lr_retrain,
                                            )
                return optim

    def mini_server(self, exp_train):
        """This funtion implements one task for the server - IAG version"""
        print(f"(PRINT) Task id {self.task_id}, training on {exp_train.classes_in_this_experience}")
        cl_data_indices = sampler(exp_train.dataset, n_clients=self.num_clients, n_classes=exp_train.classes_in_this_experience, alpha=self.args.alpha)
        

        save_train_losses = []
        save_train_accs = []
        save_test_losses = []
        save_test_accs = []
        train_losses, train_accs = [], []
        test_losses,test_accs = [], []
        self.client_gamma = {i: [] for i in range(self.num_clients)}

        client_datasets = []
        for cl_id in range(self.num_clients):
            sub_exp_train = copy.deepcopy(exp_train)
            sub_exp_train.dataset = sub_exp_train.dataset.subset(cl_data_indices[cl_id])
            client_datasets.append(sub_exp_train)
        for epoch in tqdm(range(self.args.n_rounds),):
            self.global_epoch = epoch
            clients_losses, clients_accs = [], []
            clients_test_losses, clients_test_accs = [], []
            idx_clients = [i for i in range(self.num_clients)]
            
            if self.IAG_flag:
                iag_grads_client_list = [self._calc_iag_grads(cl_idx, client_datasets[cl_idx]) for cl_idx in range(self.num_clients)]
                grads_iag_dict_server = self._process_iag_grads(iag_grads_client_list,)
            
            elif self.proxy_adaptive_lr:
                iag_grads_client_list = [self._calc_iag_grads(cl_idx, client_datasets[cl_idx]) for cl_idx in range(self.num_clients)]
                grads_iag_dict_server = self._process_iag_grads(iag_grads_client_list,)
            
            else:
                iag_grads_client_list = [None]*self.num_clients
                grads_iag_dict_server = None
            
            if (self.grad_f_flag) and (self.task_id > 0):
                if self.kick_memory and self.IAG_flag:
                    grads_g_full = self._get_full_grads_g(grads_iag_dict_server)
                    grads_f_clients_list_samples = [self._get_memory_grads_new(cl_idx,grads_g=grads_g_full) for cl_idx in range(self.num_clients)]
                    grads_f_clients_list = [cl[0] for cl in grads_f_clients_list_samples]
                    elig_clients = sum([1 if cl[1]>0 else 0 for cl in grads_f_clients_list_samples])
                else:
                    grads_f_clients_list_samples = [self._get_memory_grads_new(cl_idx,grads_g=None) for cl_idx in range(self.num_clients)]
                    grads_f_clients_list = [cl[0] for cl in grads_f_clients_list_samples]
                    elig_clients = sum([1 if cl[1]>0 else 0 for cl in grads_f_clients_list_samples])
                grads_f_server_dict = {k: sum(d[k] for d in grads_f_clients_list) for k in grads_f_clients_list[0]}
                # Average them out
                # grads_f_server_dict = {k: torch.div(v, len(grads_f_clients_list)) for k, v in grads_f_server_dict.items()}
                grads_f_server_dict = {k: torch.div(v, max(1,elig_clients)) for k, v in grads_f_server_dict.items()}
        
            else:
                grads_f_clients_list, grads_f_server_dict = None, None
            
            if (self.proxy_adaptive_lr) and (self.task_id > 0):
                    current_lr = torch.tensor(self.args.lr_retrain)
                    self.alp_proxy, self.bet_proxy = self._get_proxy_adaptive_lr(grads_f_server_dict, grads_iag_dict_server, current_lr)
            else:
                current_lr = torch.tensor(self.args.lr)
                self.alp_proxy, self.bet_proxy = torch.tensor(self.args.lr), torch.tensor(self.args.lr)
            
            
            for cl_id in idx_clients:
                cl_loss, cl_acc = self._train_client_adap_new(cl_id, client_datasets[cl_id], grads_iag_dict_server, iag_grads_client_list[cl_id], grads_f_server_dict, )
                clients_losses.append(cl_loss)
                clients_accs.append(cl_acc)
                cl_test_loss, cl_test_acc = self.test(model=self.cl_models_list[cl_id])
                clients_test_losses.append(cl_test_loss)
                clients_test_accs.append(cl_test_acc)
                # print(f"From mini-server Task: {self.task_id}, client: {cl_id}, Server round: {epoch} training loss: {cl_loss}, training accuracy: {cl_acc}")
                # print(f"From mini-server Task: {self.task_id}, client: {cl_id}, Server round: {epoch} test loss: {cl_test_loss}, test accuracy: {cl_test_acc}")
                

            
            # Update server model based on client models and then transmit to clients
            updated_weights = average_weights(self.cl_models_list, state_dict=False)
            self.global_model.load_state_dict(updated_weights)
            for cl_id in range(self.num_clients):
                self.cl_models_list[cl_id].load_state_dict(updated_weights)
            

            # Train part
            avg_loss = sum(clients_losses)/ len(clients_losses)
            avg_acc = sum(clients_accs)/ len(clients_accs)
            train_losses.append(avg_loss)
            train_accs.append(avg_acc)
            # Test part
            avg_test_loss = sum(clients_test_losses)/ len(clients_test_losses)
            avg_test_acc = sum(clients_test_accs)/ len(clients_test_accs)
            test_losses.append(avg_test_loss)
            test_accs.append(avg_test_acc)
            

            save_train_accs.append(clients_accs)
            save_train_losses.append(clients_losses)
            save_test_accs.append(clients_test_accs)
            save_test_losses.append(clients_test_losses)
            
        
        print(f"At the end of task {self.task_id} length of memory buffers: {[self.client_buffer_size[client_idx] for client_idx in range(self.num_clients)]}")
        ## Update memory buffer for each client
        for cl_id in idx_clients:
            
            self.memory_buffer_updator(cl_id, client_datasets[cl_id], name=self.args.memory_scheme)
        print(f"\nResults after Task: {self.task_id}, Epoch: {epoch + 1} global rounds of training:")
        print(f"---> Avg Training Loss, accuracy (before aggregation): {sum(train_losses) / len(train_losses), sum(train_accs) / len(train_accs)}")
        print(f"---> Avg Test Loss, accuracy (before aggregation): {sum(test_losses) / len(test_losses), sum(test_accs) / len(test_accs)}")
        
        with open(result_path+'Train_loss_task'+str(self.task_id)+'.pkl', 'wb') as f:
            pickle.dump(save_train_losses, f)
        with open(result_path+'Train_accuracy_task'+str(self.task_id)+'.pkl', 'wb') as f:
            pickle.dump(save_train_accs, f)
        with open(result_path+'Test_loss_task'+str(self.task_id)+'.pkl', 'wb') as f:
            pickle.dump(save_test_losses, f)
        with open(result_path+'Test_accuracy_task'+str(self.task_id)+'.pkl', 'wb') as f:
            pickle.dump(save_test_accs, f)

        
        test_loss, test_acc = self.test()
        print(f"After full training on task (after aggregation, global model) {self.task_id}, Test Loss: {test_loss}, Test Accuracy: {test_acc}")

        # Get the training accuracies till the end of the task
        train_task_metric_dict = {task_id: self.train_metric_with_task_id(task_id) for task_id in range(self.task_id + 1) }
        print(f"At the end of task_id: {self.task_id} train metrics(global model) are as follows (loss,acc): \n {train_task_metric_dict}")
        
        task_metric_dict = {task_id: self.test_with_task_id(task_id) for task_id in range(self.task_id + 1) }
        print(f"At the end of task_id: {self.task_id} test metrics(global model) are as follows (loss,acc): \n {task_metric_dict}")
        self.task_store.append(task_metric_dict)
        # # Saving results
        
        with open(global_result_path+'test_accuracy_task'+str(self.task_id)+'.pkl', 'wb') as f:
            pickle.dump(task_metric_dict, f)
        print(f"Task {self.task_id} completed")
        print("******####"*10)

    def _calc_iag_grads(self, client_idx, sub_exp_train, model=None,):
        """Calculate the full grads in  batched/full form for the global model - IAG method
        Returns:
        grads_list: A list of tuples containing the name of the parameter and the gradient. If batched, one per batch
        """
        if model == None:
            model = copy.deepcopy(self.cl_models_list[client_idx])
        else:
            model = copy.deepcopy(model).to(self.device)
        # model.train()
        optimizer = self.create_optimizer(self.task_id, model)
        
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        
        
        train_loader, _ = self._get_dataloader(sub_exp_train.dataset, only_train=True, shuffle=self.train_shuffle)
        task_id_device = torch.tensor(copy.copy(self.task_id)).to(self.device)
        
        if self.IAG_batch_flag:
            model.train()
            grads_dict = {}
            for idx, (data, target, task_id_label) in enumerate(train_loader):
                target = target % self.class_per_head
                # print(target)
                data, target, task_id_label = data.to(self.device), target.to(self.device), task_id_label.to(self.device)
                if self.multi_head_flag:
                    logits = model(data, task_id_device)
                else:
                    logits = model(data, 0)
                loss = criterion(logits, target)
                # if loss.item() > 10.0:
                #     print(f"loss inside calc_iag grads 1 is: {loss.item()}")
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.norm_threshold)
                with torch.no_grad():
                    
                    grads_batch = {str(name): param.grad.clone().detach() if param.grad is not None else torch.zeros_like(param) for (name, param) in model.named_parameters()}
                
                optimizer.zero_grad()
                grads_dict[str(idx)] = grads_batch
        else:
            model.train()
            grads_dict = {}
            for idx, (data, target, task_id_label) in enumerate(train_loader):
                target = target % self.class_per_head
                data, target, task_id_label = data.to(self.device), target.to(self.device), task_id_label.to(self.device)
                if self.multi_head_flag:
                    logits = model(data, task_id_device)
                else:
                    logits = model(data, 0)
                loss = criterion(logits, target)
                loss.backward()
                
            with torch.no_grad():
                
                grads_full = {str(name): torch.div(param.grad.clone().detach(),len(train_loader)) if param.grad is not None else torch.zeros_like(param) for (name, param) in model.named_parameters()}
            
            optimizer.zero_grad()
            grads_dict["full"] = grads_full
        
        del model, optimizer, criterion, train_loader, task_id_device, data, target, task_id_label, logits, loss
        
        
        return grads_dict
    
    def _get_full_grads_g(self, grads_iag_dict_server):
        if grads_iag_dict_server is None:
            return None
        
        if self.IAG_batch_flag:
            grads_g_summed = dict(sum(map(Counter(), grads_iag_dict_server.values()), Counter()))
            grads_g = [torch.div(grad, len(list(grads_iag_dict_server.values()))) for name, grad in grads_g_summed.items()]

            del grads_g_summed
        else:
            grads_g_dict = grads_iag_dict_server["full"]
            grads_g = [grad for name, grad in grads_g_dict.items()]
            del grads_g_dict
        
        
        return grads_g

    def _get_proxy_adaptive_lr(self, grads_f_server_dict, grads_iag_dict_server, current_lr,):
        """Get the adaptive learning rate for the server"""
        grads_f = [grad for name, grad in grads_f_server_dict.items()]
        if grads_f_server_dict is not None:
            grads_g = self._get_full_grads_g(grads_iag_dict_server)
        else:
            raise ValueError("grads_f_server_dict is None")
        
        
        inn = self._inner_prod_sum(grads_f, grads_g)
        
        norm_f_2 = self._inner_prod_sum(grads_f, grads_f) 
        norm_g_2 = self._inner_prod_sum(grads_g, grads_g)
        print(f"Inner product for proxy is: {inn} for norm_f {norm_f_2}, norm_g {norm_g_2}")
        
        if inn <= 0:
            alp_proxy = current_lr*(1- torch.div(inn,max(norm_f_2, self.eps)))
            # alp_proxy = torch.clamp(alp_proxy, min=1e-4, max=self.max_clip)
            beta_proxy = current_lr 
            if abs(alp_proxy) > 2/self.L:
                print(f"Alp_proxy is more than 2/L with value: {alp_proxy}")
                # stop
            if abs(beta_proxy) > 2/self.L:
                print(f"Beta_proxy is more than 2/L with value: {beta_proxy}")
                # stop
        else:
            alp_proxy = current_lr 
            inn = torch.clamp(inn, min=self.eps,)
            beta_proxy = ((1- current_lr*self.L) / max((self.L * norm_g_2), self.eps))* inn
            

        del grads_f, grads_g, inn, norm_f_2, norm_g_2
        # gc.collect()
        return alp_proxy, beta_proxy
    
    
    def accuracy_new(y_pred, y_true):
        assert len(y_pred) == len(y_true), "Data length error."
        all_acc = {}
        all_acc["total"] = np.around(
            (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
        )
        return all_acc

    def _process_iag_grads(self, iag_grads_dict_list:list,):
        
        
        if self.IAG_batch_flag:
            start = time.time()
            all_batch_ids = set().union(*iag_grads_dict_list)
            batch_dicts = {batch_id: [client_dict[batch_id] for client_dict in iag_grads_dict_list if batch_id in client_dict.keys()] for batch_id in all_batch_ids }
            param_keys_init = iag_grads_dict_list[0][str(0)].keys() #model param names
            grads_iag_dict =  {batch_id: {k: sum(d[k] for d in batch_dicts[batch_id]) for k in param_keys_init} for batch_id in batch_dicts.keys()}
            grads_iag_dict = {batch_id: {k: torch.div(v, len(batch_dicts[batch_id])) for k, v in grads_iag_dict[batch_id].items()} for batch_id in grads_iag_dict.keys()}
            
            
            del all_batch_ids, batch_dicts, param_keys_init, 
        else:
            start = time.time()
            iag_grads_dict_list = [d["full"] for d in iag_grads_dict_list if "full" in d.keys()]
            grads_iag_dict = {k: sum(d[k] for d in iag_grads_dict_list) for k in iag_grads_dict_list[0]}
            # grads_iag_dict = dict(sum(map(Counter, iag_grads_dict_list), Counter())) # ISSUE (fixed below) (25 Sep'24): not dividing by number of clients
            grads_iag_dict = {"full": {k : torch.div(v, len(iag_grads_dict_list)) for k, v in grads_iag_dict.items()}}
            

            del iag_grads_dict_list
        
        return grads_iag_dict

    def _normalize_grad(self, model, grad_list):
        """Mutates the "grad_list" in-place"""
        # Accumulate and normalize gradients
        for name, param in model.named_parameters():
            if (param.grad is not None) and ("task_classifiers" not in name):
                grad = copy.deepcopy((param.grad))
                old_grad = grad_list[name]
                old_grad += grad
                norm = old_grad.norm()
                if norm != 0:  # Avoid division by zero
                    normalized_grad = old_grad / norm
                    grad_list[name] = normalized_grad
                
    
    def _get_client_memory_grads_old(self, client_idx):
        """Compute grads on the memory data and return the grads"""
        # model = copy.deepcopy(self.global_model)
        model = self.cl_models_list[client_idx]
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = self.opt_lists[client_idx]
        # Get the train loader
        storage_p = self.memory_buffer[client_idx]
        train_loader_memory, _ = self._get_dataloader(storage_p.buffer, batch_size=1, only_train=True) #batch_size=1
        
        grads_f = {name: torch.zeros_like(param) for name, param in model.named_parameters() if 'task_classifiers' not in name}
        grads_f_full = [(name, param.grad.clone().detach()) if (param.grad is not None) else (name, torch.zeros_like(param)) for (name, param) in model.named_parameters() ]
        
        
        grads_f_full = [grad for (name, grad) in grads_f_full if 'task_classifiers' not in name]
        
        samples = 0
        for idx, (data, target, task_id_label) in enumerate(train_loader_memory):
            if idx > 5:
                # break
                pass
            target = target % self.class_per_head
            data, target, task_id_label = data.to(self.device), target.to(self.device), task_id_label.to(self.device)
            optimizer.zero_grad()
            logits = model(data, task_id_label, memory_flag=True)
            
            loss = criterion(logits, target)
            loss.backward()
            pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.norm_threshold)
            with torch.no_grad():
                model_params = dict(model.named_parameters())
                model_params_list = [param for (name, param) in model.named_parameters() if "task_classifiers" not in name]
                grads_f_full = [param.grad.clone().detach() + grad_f if (param.grad is not None) else grad_f 
                                for (param, grad_f) in zip(model_params_list, grads_f_full)]


                
                _ = self._normalize_grad(model, grads_f)

                samples += data.size(0)
        grads_f = {name: grad / len(train_loader_memory) for (name, grad) in grads_f.items() if "task_classifiers" not in name}
        # grads_f_full = [grad/len(train_loader_memory) for (name, grad) in grads_f_full.values()]
        grads_f_full = [grad/len(train_loader_memory) for grad in grads_f_full]
        optimizer.zero_grad()
        return list(grads_f.values()), grads_f_full
    
    def _get_memory_grads_new(self, client_idx, grads_g:list=None):
        
        model = copy.deepcopy(self.cl_models_list[client_idx]).to(self.device)
        
        optimizer = self.create_optimizer(self.task_id, model)
        
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        
        storage_p_all = self.memory_buffer[client_idx]
        storage_p_list = [v for v in storage_p_all.buffer_groups.values()]
        mem_batch_size = 16
        num_samples = 0
        for buf_id, storage_p in enumerate(storage_p_list):
            sample_size = min(min(self.memory_sample_size, len(storage_p.buffer)), len(storage_p.buffer))
            sample_idx = random.sample(range(len(storage_p.buffer)), sample_size)
            memory_data = storage_p.buffer.subset(sample_idx)
            print(f"Memory buffer size for client {client_idx}: buffer id: {buf_id}, {len(storage_p.buffer)}, sampled size: {sample_size}")
            
            train_loader_memory, _ = self._get_dataloader(memory_data, batch_size=mem_batch_size, only_train=True) 
            # sampled_targets = []
                    
            model.train()
            for idx, (data, target, task_id_label) in enumerate(train_loader_memory):
                task_id_device = torch.tensor(copy.copy(buf_id)).to(self.device) 
                
                target = target % self.class_per_head
                data, target, task_id_label = data.to(self.device), target.to(self.device), task_id_label.to(self.device)
                if self.multi_head_flag:
                    
                    logits = model(data, task_id_device,)
                else:
                    logits = model(data, 0,)
                loss = criterion(logits, target)
                
                
                
                if grads_g is not None:
                    grads_f = torch.autograd.grad(loss, model.parameters(), create_graph=False, allow_unused=True, retain_graph=True)
                    with torch.no_grad():
                        
                        grads_f = [grads_f[i].clone().detach() if grads_f[i] is not None else torch.zeros_like(param) for i, (name,param) in enumerate(model.named_parameters())]
                        inn = self._inner_prod_sum(grads_f, grads_g)
                    if inn >= self.eps:
                        loss.backward() 
                        num_samples += data.shape[0]
                    else:
                        continue 
                else: 
                    loss.backward()
                    num_samples += data.shape[0]
                
        if num_samples > 0: 
            grads_f_client = {str(name): torch.div(param.grad.clone().detach(), num_samples) if param.grad is not None else torch.zeros_like(param) for (name, param) in model.named_parameters()}
        else: 
            grads_f_client = {str(name): torch.zeros_like(param) for (name, param) in model.named_parameters()} 
        optimizer.zero_grad()

        
        print(f"Client:{client_idx} sampled {num_samples} from memory for kick_memory {self.kick_memory}")
        
        del model, optimizer, criterion, train_loader_memory, task_id_device, data, target, task_id_label, logits, loss
        del sample_size, sample_idx, buf_id
        # gc.collect()
        return (grads_f_client, num_samples)
    
    def _inner_prod_sum(self, grad_list_1, grad_list_2):
        """Calculate inner product"""
        inn = torch.tensor(0.0).to(self.device)
        for grad1, grad2 in zip(grad_list_1, grad_list_2):
            
            product = torch.div(torch.dot(grad1.view(-1), grad2.view(-1)), len(grad1.view(-1))) # Try this instead of lst divison
            inn += product
        del product
        # gc.collect()
        return inn
        
    
    
    
    def _sigmoid(self, x):
        return 1/ (1 + math.exp(-x))
    
    def _fetch_memory_data(self, client_idx):
        """Returns data for training from the memory buffer"""
        storage_p = self.memory_buffer[client_idx]
        memory_data = storage_p.buffer
        return memory_data

    
    def calc_new_grads_adaptive_batch(self, grads_f:list, grads_g:list, current_lr, alpha_lr_proxy=None, beta_lr_proxy=None):
        """For Every Step Adaptive Learning
        Returns:
        new_grads: list of new gradients
        """
        if alpha_lr_proxy is None:
            alpha_lr_proxy = self.alp_proxy
        if beta_lr_proxy is None:
            beta_lr_proxy = self.bet_proxy
        alp_t, beta_t = alpha_lr_proxy, beta_lr_proxy
        
        
        inn = self._inner_prod_sum(grads_f, grads_g)
        norm_f_2 = self._inner_prod_sum(grads_f, grads_f)
        norm_g_2 = self._inner_prod_sum(grads_g, grads_g)
        
        if inn >0: # To avoid very small beta
            inn = torch.clamp(inn, min=self.eps,)
        # divide by learning_rate for further use by optimizer
        beta = ((1-alp_t*self.L) / max(self.L * norm_g_2, self.eps)) * inn # ISSUE (24 Sep'24): check correct default_lr
        # clamp values
        beta = torch.clamp(beta, min=self.eps,) 
        
        
        mul_factor_1 = beta/max(current_lr,self.eps)
        mul_factor_2 = beta_t/max(current_lr,self.eps)
        
        new_grads_g = [torch.where(inn >0, mul_factor_1*grads_g[i], mul_factor_2*grads_g[i]) for i in range(len(grads_g))] 

        del inn, norm_f_2, norm_g_2, mul_factor_1, mul_factor_2, beta
        # gc.collect()
        return new_grads_g
    
    def calc_new_grads_adaptive_step(self, grads_f, grads_g, current_lr,alpha_lr_proxy=None, beta_lr_proxy=None, add_memory=False):
        
        if alpha_lr_proxy is None:
            alpha_lr_proxy = self.alp_proxy
        if beta_lr_proxy is None:
            beta_lr_proxy = self.bet_proxy
        alp_t, beta_t = alpha_lr_proxy, beta_lr_proxy
        

        inn = self._inner_prod_sum(grads_f, grads_g)
        norm_g_2 = self._inner_prod_sum(grads_g, grads_g)
        norm_f_2 = self._inner_prod_sum(grads_f, grads_f)
        print(f"Inner product (calc_new_grads_adaptive_step): {inn}, norm_f: {norm_f_2}, norm_g: {norm_g_2}")
        
        if inn >0: 
            inn = torch.clamp(inn, min=self.eps,)
        # divide by learning_rate for further use by optimizer - doing in next steps in mul_factor
        beta_t_i = (inn* (1- alp_t*self.L) / max(self.L * norm_g_2, self.eps) ) 
        if self.optim_name == "adam":
            beta_t_i = torch.clamp(beta_t_i, max=0.001, min = 0.000001)
        elif self.optim_name == "sgd":
            beta_t_i = torch.clamp(beta_t_i, max=0.95, min= 0.0001)
        

        alpha_t_i = torch.mul(alp_t, (1- torch.div(inn,max(norm_f_2, self.eps)))) 
        print("Client adaptive learning rate (step): ", alpha_t_i, beta_t_i)
        
        mul_factor_1 = (beta_t_i/max(beta_t,self.eps))* (1/current_lr)
        mul_factor_2 = 1/max(current_lr,self.eps) 
        
        
        grads_g_new = [torch.where(inn >0, torch.mul(mul_factor_1, grads_g[i]), torch.mul(mul_factor_2, grads_g[i])) for i in range(len(grads_g))] 
        if add_memory:
            
            grads_f_new = [torch.where(inn <=0, torch.mul(alpha_t_i/current_lr, grads_f[i]), torch.mul(alp_t/current_lr, grads_f[i])) for i in range(len(grads_f))]
            
            new_grads = [f+g for (f,g) in zip(grads_f_new, grads_g_new)]
            
            del grads_f_new
        else:
            new_grads = grads_g_new
        
        del inn, beta_t_i, alpha_t_i, mul_factor_1, mul_factor_2
        # gc.collect()
        return new_grads

    def _grad_updator(self, grads_list, model):
        with torch.no_grad():
            for idx, (name, param) in enumerate(model.named_parameters()):
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                param.grad = (torch.clamp(grads_list[idx].detach(), min=self.min_clip, max=self.max_clip))

    def _train_client_curr(self, client_idx, sub_exp_train, server_iag_dict:dict, client_iag_dict:dict, grads_f_dict:dict=None, adaptive:bool=False): 
        """Train a client on current task data
        Only handles the current training part - on g
        """
        task_id_device = torch.tensor(copy.copy(self.task_id)).to(self.device)
        model = self.cl_models_list[client_idx]
        if grads_f_dict is None: 
            adaptive = False
        # else:
        #     adaptive_flag = self.adaptive_flag
        model.train()
        criterion = nn.CrossEntropyLoss()
        
        optimizer = self.create_optimizer(self.task_id, model)
        lr_lmbda = lambda epoch: 1/math.sqrt(self.args.n_client_epochs) 
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lmbda, last_epoch=-1)
        # Get the train loader and test loader
        train_loader, test_loader = self._get_dataloader(sub_exp_train.dataset, shuffle=self.train_shuffle, drop_last=False)
        pass
        optimizer.zero_grad()
        
        for epoch in range(self.args.n_client_epochs):
            model.train()
            last_model_parameters = {name: param.clone().detach() for name, param in model.named_parameters()}
            for batch_id, (data, target, task_id_label) in enumerate(train_loader):
                target = target % self.class_per_head
                data, target, task_id_label = data.to(self.device), target.to(self.device), task_id_label.to(self.device)
                
                if self.multi_head_flag:
                    logits = model(data, task_id_device)
                else:
                    logits = model(data, 0)
                loss = criterion(logits, target)
                
                
                loss.backward()
                
                if (self.IAG_flag) and (self.IAG_batch_flag) and (self.delayed_grad_batch_flag): # case 3
                    # print("case 3") ## check
                    with torch.no_grad():
                        new_grads_iag = {name: param.grad.clone().detach() if param.grad is not None else torch.zeros_like(param) for (name, param) in model.named_parameters()}
                        new_grads_iag = [server_iag_dict[str(batch_id)][name]  - client_iag_dict[str(batch_id)][name] + new_grads_iag[name] for name, param in model.named_parameters()]
                        
                        if adaptive and self.adaptive_lr_batch_flag:
                            grads_f = [grad for name, grad in grads_f_dict.items()]
                            current_lr = optimizer.param_groups[-1]['lr']
                            grads_g_final = self.calc_new_grads_adaptive_batch(grads_f, new_grads_iag,current_lr=current_lr,alpha_lr_proxy=self.alp_proxy, beta_lr_proxy=self.bet_proxy) 
                            _ = self._grad_updator(grads_g_final, model)
                            
                        else:
                            _ = self._grad_updator(new_grads_iag, model)
                    optimizer.step()
                    optimizer.zero_grad()
                
                elif (self.IAG_flag) and (not self.IAG_batch_flag) and (self.delayed_grad_batch_flag): 
                    # print("case 2")
                    with torch.no_grad():
                        new_grads_iag = {name: param.grad.clone().detach() if param.grad is not None else torch.zeros_like(param) for (name, param) in model.named_parameters()}
                        new_grads_iag = [server_iag_dict["full"][name]  - client_iag_dict["full"][name] + new_grads_iag[name] for name, param in model.named_parameters()]
                                                
                        if adaptive and self.adaptive_lr_batch_flag:
                            grads_f = [grad for name, grad in grads_f_dict.items()]
                            current_lr = optimizer.param_groups[-1]['lr']
                            grads_g_final = self.calc_new_grads_adaptive_batch(grads_f, new_grads_iag,current_lr, self.alp_proxy,self.bet_proxy) 
                            _ = self._grad_updator(grads_g_final, model)
                            
                        
                        else:
                            _ = self._grad_updator(new_grads_iag, model)
                    optimizer.step()
                    optimizer.zero_grad()
                
                elif (not self.IAG_flag) and (not self.IAG_batch_flag) and (self.delayed_grad_batch_flag): # normal SGD, batch level adaptive with no IAG
                    
                    with torch.no_grad():
                        grads_g = {name: param.grad.clone().detach() if param.grad is not None else torch.zeros_like(param) for (name, param) in model.named_parameters()}
                        grads_g = [param for name,param in grads_g.items()]
                        
                        if adaptive and self.adaptive_lr_batch_flag:
                            # print("case NCCL, check train_adap outer loop adding grads_f also")
                            grads_f = [grad for name, grad in grads_f_dict.items()]
                            current_lr = optimizer.param_groups[-1]['lr']
                            # grads_g = [param for name,param in grads_g.items()]
                            grads_g_final = self.calc_new_grads_adaptive_batch(grads_f, grads_g, current_lr, self.alp_proxy, self.bet_proxy) 
                            _ = self._grad_updator(grads_g_final, model)
                            
                        else:
                            _ = self._grad_updator(grads_g, model)
                    pass 
                    optimizer.step()
                    optimizer.zero_grad()

                else: 
                    continue

            if (self.IAG_flag) and (not self.IAG_batch_flag) and (not self.delayed_grad_batch_flag): #case 1
                # print("case 1")
                with torch.no_grad():
                    new_grads_iag = {name: param.grad.clone().detach() if param.grad is not None else torch.zeros_like(param) for (name, param) in model.named_parameters()}
                    new_grads_iag = [server_iag_dict["full"][name]  - client_iag_dict["full"][name] + new_grads_iag[name] for name, param in model.named_parameters()]
                      
                    if adaptive and self.adaptive_lr_step_flag:
                        grads_f = [grad for name, grad in grads_f_dict.items()]
                        current_lr = optimizer.param_groups[-1]['lr']
                        grads_g_final = self.calc_new_grads_adaptive_step(grads_f, new_grads_iag, current_lr, self.alp_proxy, self.bet_proxy, add_memory=False)
                        _ = self._grad_updator(grads_g_final, model)
                    else:
                        _ = self._grad_updator(new_grads_iag, model)
                optimizer.step()
                optimizer.zero_grad()
            
            elif (self.adaptive_lr_step_flag) and (adaptive):
               
                a_i = [last_model_parameters[name] - param.clone().detach() for (name, param) in model.named_parameters()] 
                assert grads_f_dict is not None, "grads_f is None in the cflag proposed training"
                grads_f = [grad for name, grad in grads_f_dict.items()] 
                with torch.no_grad():
                    current_lr = optimizer.param_groups[-1]['lr']            
                    new_grads_g = self.calc_new_grads_adaptive_step(grads_f, a_i, current_lr=current_lr, alpha_lr_proxy=self.alp_proxy, beta_lr_proxy=self.bet_proxy, add_memory=False)
                    _ = self._grad_updator(new_grads_g, model)
                    
                optimizer.step()
                optimizer.zero_grad()
            
            # Scheduler step
            if self.task_id >0:
                pass
                # scheduler.step()
            
            train_task_metric_dict = {task_id: self.train_metric_with_task_id(task_id, model=model) for task_id in range(self.task_id+1) }
            task_metric_dict = {task_id: self.test_with_task_id(task_id, model=model) for task_id in range(self.task_id+1) }
            

        #save results
        with open(client_result_path+'test_accuracy_model'+str(client_idx)+'task'+str(self.task_id)+'.pkl', 'wb') as f:
             pickle.dump(task_metric_dict, f)

        
        print(f"--------"*20)

        try:
            del task_id_device, optimizer, criterion, train_loader, test_loader, last_model_parameters
        except:
            pass
        # gc.collect()
    
    def memory_buffer_updator(self, client_idx, sub_exp_train, name=None ):
        """Function to update memory buffer"""
        
        strategy_state = SimpleNamespace(experience=sub_exp_train)
        storage_p = self.memory_buffer[client_idx]
        if self.task_id >0:
            self.client_buffer_size[client_idx] += self.args.initial_buffer_size
            storage_p.resize(strategy_state, self.client_buffer_size[client_idx])
        _ = storage_p.update(strategy_state)
    
    def _train_client_adap(self,
                client_idx,
                sub_exp_train, server_iag_dict:dict,
                client_iag_dict:dict, grads_f_server:dict=None,
                # adaptive:bool=False
                ):
        
        """Trains a client for one global server round on both the memory data and the current task data"""
        if (self.task_id == 0) or (not self.grad_f_flag):
            _ = self._train_client_curr(client_idx, sub_exp_train, server_iag_dict=server_iag_dict, client_iag_dict=client_iag_dict, grads_f_dict=None, adaptive=False)       
        elif (self.task_id > 0) and (grads_f_server is not None): # ISSUE: verify this logic, using iag during training 
            
            if self.adaptive_lr_round_flag:
                
                _ = self._train_client_curr(client_idx, sub_exp_train, server_iag_dict=server_iag_dict, client_iag_dict=client_iag_dict, grads_f_dict=None, adaptive=False)
            else:
                _ = self._train_client_curr(client_idx, sub_exp_train, server_iag_dict=server_iag_dict, client_iag_dict=client_iag_dict, grads_f_dict=grads_f_server, adaptive=self.adaptive_flag)
            
            model = self.cl_models_list[client_idx] # load model here after training only
            model.train()
            with torch.no_grad():
                global_model_parameters = {name: param.clone().detach() for name, param in self.global_model.named_parameters()}
                a_i = [global_model_parameters[name] - param_client.clone().detach() for (name, param_client) in self.cl_models_list[client_idx].named_parameters()] # cumulative grads
                assert grads_f_server is not None, "grads_f is None in the cflag proposed training"
                grads_f = [grad for name, grad in grads_f_server.items()] ## ISSUE (27 Sep'24): what if grads_f is none, Also ensure grads f without adaptive is performed

                
                optimizer = self.create_optimizer(self.task_id, model)
                current_lr = optimizer.param_groups[-1]['lr']
                if self.adaptive_lr_round_flag:
                    new_grads = self.calc_new_grads_adaptive_step(grads_f, a_i, current_lr=current_lr, alpha_lr_proxy=self.alp_proxy, beta_lr_proxy=self.bet_proxy, add_memory=True)
                    _ = self._grad_updator(new_grads, model)
                    
                else:
                    pass 
                    
                    new_grads = [f + g for (f,g) in zip(grads_f, a_i)]
                    _ = self._grad_updator(new_grads, model)
                try:
                    del a_i, grads_f, global_model_parameters
                except:
                    pass
            optimizer.step()
            optimizer.zero_grad()
            

        ## Fetch scores
        train_loader, _ = self._get_dataloader(sub_exp_train.dataset, only_train=True)
        train_loss, train_acc = self.test(train_loader, model=self.cl_models_list[client_idx])
        
        train_task_metric_dict = {task_id: self.train_metric_with_task_id(task_id, model=self.cl_models_list[client_idx]) for task_id in range(self.task_id+1) }
        task_metric_dict = {task_id: self.test_with_task_id(task_id, model=self.cl_models_list[client_idx]) for task_id in range(self.task_id+1) }
        formatted_result_train, formatted_results = format_dict_with_precision(train_task_metric_dict), format_dict_with_precision(task_metric_dict)
        
        try:
            del train_task_metric_dict, task_metric_dict, global_model_parameters, a_i, grads_f, current_lr, new_grads, optimizer
        except:
            print("OLD Unable to delete in train_adap")
        # gc.collect()
        return train_loss, train_acc
    
    def _train_client_adap_new(self,
                client_idx,
                sub_exp_train, server_iag_dict:dict,
                client_iag_dict:dict, grads_f_server:dict=None,
                
                ):
        
        """Trains a client for one global server round on both the memory data and the current task data"""
        if (self.task_id == 0) or (not self.grad_f_flag):
            _ = self._train_client_curr(client_idx, sub_exp_train, server_iag_dict=server_iag_dict, client_iag_dict=client_iag_dict, grads_f_dict=None, adaptive=False)       
        elif (self.task_id > 0) and (grads_f_server is not None):
            
            _ = self._train_client_curr(client_idx, sub_exp_train, server_iag_dict=server_iag_dict, client_iag_dict=client_iag_dict, grads_f_dict=grads_f_server, adaptive=self.adaptive_flag)
            
            model = self.cl_models_list[client_idx] # load model here after training only
            model.train()
            optimizer = self.create_optimizer(self.task_id, model)
            # Train on memory data
            # Get grad g
            grads_g_cl = [self._calc_iag_grads(client_idx, sub_exp_train)]
            grads_g_dict = self._process_iag_grads(grads_g_cl)
            grads_g = self._get_full_grads_g(grads_g_dict)
            # Take memory grads
            if self.kick_memory:
                grads_f_dict, num_samples = self._get_memory_grads_new(client_idx,grads_g=grads_g)
            else:
                grads_f_dict, num_samples = self._get_memory_grads_new(client_idx,grads_g=None)
            grads_f_new = [param for name, param in grads_f_dict.items()] 

            with torch.no_grad():
                
                _ = self._grad_updator(grads_f_new, model)
            optimizer.step()
            optimizer.zero_grad()
            del optimizer

            global_model_parameters = {name: param.clone().detach() for name, param in self.global_model.named_parameters()}
            ai = [global_model_parameters[name] - param_client.clone().detach() for (name, param_client) in model.named_parameters()] # cumulative grads
            inn = self._inner_prod_sum(grads_f_new, ai)
            gamma_i = self._calc_gamma(self.bet_proxy, ai, self.alp_proxy, inn)
            self.client_gamma[client_idx].append(gamma_i.item())
            
        
        ## Fetch scores
        train_loader, _ = self._get_dataloader(sub_exp_train.dataset, only_train=True)
        train_loss, train_acc = self.test(train_loader, model=self.cl_models_list[client_idx])
        
        task_metric_dict = {task_id: self.test_with_task_id(task_id, model=self.cl_models_list[client_idx]) for task_id in range(self.task_id+1) }
        formatted_results = format_dict_with_precision(task_metric_dict)
        print(f"(E local, 1 memory) Task: {self.task_id}, client: {client_idx}, Server epoch: {self.global_epoch}, test_stats (f+g) on local model: {formatted_results}")
        

        try:
            del train_task_metric_dict, task_metric_dict
        except:
            print("Unable to delete in train_adap new")
        gc.collect()
        return train_loss, train_acc

    def _get_dataloader(self, training_dataset, batch_size=None, only_train=False, shuffle=None, drop_last=None):
        """Retuns dataloader"""
        if batch_size == None:
            batch_size = self.args.batch_size
        if shuffle is None:
            shuffle = True
        if drop_last is None:
            drop_last = False
        # print(f"Dataset size: {len(training_dataset)}")
        train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        if not only_train:
            test_set = self.bm.test_stream[self.task_id].dataset
            test_loader = DataLoader(test_set, batch_size=128)
        else:
            test_loader = None
        
        return train_loader, test_loader
    
    def test(self, test_loader=None, model=None, task_id=None) -> Tuple[float, float]:
        """Test on the model. If no model passed as an argument then test on the server model."""
        if model == None:
            model = copy.deepcopy(self.global_model).to(self.device)
        model.eval()
        criterion = nn.CrossEntropyLoss()
        if test_loader == None:
            test_set = self.bm.test_stream[self.task_id].dataset
            test_loader = DataLoader(test_set, batch_size=128)
        if task_id == None:
            task_id_device = torch.tensor(copy.copy(self.task_id)).to(self.device)
        # test_loader = 
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for idx, (data, target, task_id_label) in enumerate(test_loader):
                target = target % self.class_per_head
                data, target = data.to(self.device), target.to(self.device)
                if self.multi_head_flag:
                    logits = model(data, task_id_device)
                else:
                    logits = model(data, 0)
                loss = criterion(logits, target)

                total_loss += loss.item()
                total_correct += (logits.argmax(dim=1) == target).sum().item()
                total_samples += data.size(0)
        # calculate average accuracy and loss
        total_loss /= (idx+1)
        total_acc = total_correct / total_samples

        return total_loss, total_acc
    
    def test_with_task_id(self,task_id, model=None) -> Tuple[float, float]:
        task_id_device = copy.copy(torch.tensor(task_id)).to(self.device)
        if model ==  None:
            model = copy.deepcopy(self.global_model).to(self.device)
        model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        # Fetch the test_loader based on task_id
        test_set = self.bm.test_stream[task_id].dataset
        test_loader = DataLoader(test_set, batch_size=128)
        with torch.no_grad():
            for idx, (data, target, task_id_label) in enumerate(test_loader):
                target = target % self.class_per_head
                # print(f"Target inside test_with_task_id is {target}")
                data, target, task_id_label = data.to(self.device), target.to(self.device), task_id_label.to(self.device)
                # print(f"task_id_device: {task_id_device}, task_id_label: {task_id_label}")
                if self.multi_head_flag:
                    logits = model(data, task_id_device)
                else:
                    logits = model(data, 0)
                loss = criterion(logits, target)
    
                total_loss += loss.item()
                total_correct += (logits.argmax(dim=1) == target).sum().item()
                total_samples += data.size(0)

        total_loss /= (idx+1)
        total_acc = total_correct / total_samples

        return total_loss, total_acc
    
    def train_metric_with_task_id(self, task_id, model=None) -> Tuple[float, float]:
        task_id_device = copy.copy(torch.tensor(task_id)).to(self.device)
        if model ==  None:
            model = copy.deepcopy(self.global_model).to(self.device)
        model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        # Fetch the train_loader based on task_id
        train_set = self.bm.train_stream[task_id].dataset
        train_loader = DataLoader(train_set, batch_size=128)
        with torch.no_grad():
            for idx, (data, target, task_id_label) in enumerate(train_loader):
                target = target % self.class_per_head
                data, target, task_id_label = data.to(self.device), target.to(self.device), task_id_label.to(self.device)
                if self.multi_head_flag:
                    logits = model(data, task_id_device)
                else:
                    logits = model(data, 0)
                loss = criterion(logits, target)
    
                total_loss += loss.item()
                total_correct += (logits.argmax(dim=1) == target).sum().item()
                total_samples += data.size(0)

        total_loss /= (idx+1)
        total_acc = total_correct / total_samples

        return total_loss, total_acc

    def normalize_grad_norms(self, grad:torch.tensor):
        """Clip the norms of the gradients"""
        norm = torch.linalg.norm(grad, dim=1, ord = 2)
        grad = torch.div(grad, norm)
        return grad


if __name__== "__main__":
    parser = arg_parser()

    result_path = f'./saved_results/results_{parser.dataset}_{parser.model_name}/buffer_{parser.initial_buffer_size}/alpha_{parser.alpha}/lrs_{parser.lr}-{parser.lr_retrain}/momen_{parser.momentum}-{parser.momentum_retrain}/seed_{parser.seed}/'
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    global_result_path = result_path + 'global_model/'
    if os.path.exists(global_result_path):
        shutil.rmtree(global_result_path)
    os.makedirs(global_result_path)

    client_result_path = result_path + 'local_model/'
    if os.path.exists(client_result_path):
        shutil.rmtree(client_result_path)
    os.makedirs(client_result_path)
    

    if parser.dataset == "MNIST":
        num_tasks = parser.num_tasks
        transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))  # Normalize inputs
                                         ])
        bm = SplitMNIST(n_experiences=num_tasks,
                    return_task_id=True,
                    seed=parser.seed,
                    train_transform=transform, eval_transform=transform
                    )
    elif parser.dataset == "PermutedMNIST":
        num_tasks = parser.num_tasks
        transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))  # Normalize inputs
                                        ])
        bm = PermutedMNIST(n_experiences=num_tasks,
                    return_task_id=True,
                    seed=parser.seed,
                    train_transform=transform, eval_transform=transform
                    )
    elif parser.dataset == "CIFAR10":
        num_tasks = parser.num_tasks
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        bm = SplitCIFAR10(n_experiences=num_tasks,
                    return_task_id=True,
                    seed=parser.seed,
                    train_transform=transform_train, eval_transform=transform_test
                    )
    elif parser.dataset == "CIFAR100":
        num_tasks = parser.num_tasks
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),                # Convert to tensor
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), # Normalize with CIFAR-100 stats
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        bm = SplitCIFAR100(n_experiences=num_tasks,
                    return_task_id=True,
                    seed=parser.seed,
                    train_transform=transform_train, eval_transform=transform_test
                    )
    elif parser.dataset == "tinyimagenet":
        num_tasks = parser.num_tasks
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224), # Random crop with padding
            transforms.RandomHorizontalFlip(),    # Randomly flip images horizontally
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),            # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        bm = SplitTinyImageNet(n_experiences=num_tasks,
                    return_task_id=True,
                    seed=parser.seed,
                    train_transform=transform_train, eval_transform=transform_test
                    )
    else:
        raise NotImplementedError
    print(f"Dataset: {parser.dataset} with splits: {num_tasks}")
    print(parser)
    algo = CFLAG(parser, bm, num_tasks,)
    algo.server()
    print(parser)

