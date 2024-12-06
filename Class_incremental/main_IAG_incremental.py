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
from utils_inc import arg_parser, average_weights, get_acc_forgetting
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

from models.resnet_inc import ResNet18, ResNet50

# save gamma
from functools import wraps
def time_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in: {end_time - start_time:.4f} seconds")
        return result
    return wrapper
def time_all_methods(cls):
    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value):
            setattr(cls, attr_name, time_function(attr_value))
    return cls

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

class MLP_inc(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_classes: int, num_tasks=5,):
        super(MLP_inc, self).__init__()
        # number of classes should be divisible by number of tasks
        assert n_classes % num_tasks == 0, "Number of classes should be divisible by number of tasks"
        self.num_classes_total = n_classes
        self.num_tasks = num_tasks
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.task_classifiers = nn.Linear(hidden_size, self.num_classes_total//self.num_tasks)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.out_dim = hidden_size
    
    def generate_task_classifiers(self, task_id):
        num_out = self.task_classifiers.out_features + (self.num_classes_total//(self.num_tasks))
        # num_out = (self.num_classes_total//(self.num_tasks))*(task_id+1)
        # print(f"updated cl nodes: {num_out}")
        fc_layer = nn.Linear(self.out_dim, num_out)
        return fc_layer
    
    def update_task_classifiers(self, task_id):
        fc = self.generate_task_classifiers(task_id)
        out_features = self.task_classifiers.out_features
        weight = copy.deepcopy(self.task_classifiers.weight.data)
        bias = copy.deepcopy(self.task_classifiers.bias.data)
        fc.weight.data[:out_features] = copy.deepcopy(weight)
        fc.bias.data[:out_features] = copy.deepcopy(bias)

        fc.weight.data[out_features:] = torch.randn_like(fc.weight.data[out_features:]) * 0.01
        fc.bias.data[out_features:] = torch.zeros_like(fc.bias.data[out_features:])

        
        del self.task_classifiers
        self.task_classifiers = fc
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """pass_head: provision to skip head during forward and backward pass - potential issue is loss grad calculation """
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.task_classifiers(x)
        return x

def format_dict_with_precision(data, precision=4):
    return {key: tuple(f'{val:.{precision}f}' for val in value) for key, value in data.items()}

# @time_all_methods
class CFLAG:
    """Proposed algorithm with IAG"""

    def __init__(self, args: Dict[str, Any], benchmark, num_tasks,):
        self.args = args
        self.bm = benchmark
        self.num_tasks = num_tasks
        if self.args.dataset == "CIFAR100":
            self.num_classes_total = 100
        elif self.args.dataset == "tinyimagenet":
            self.num_classes_total = 200
        else:
            self.num_classes_total = args.num_classes_total
        self.class_per_task = self.num_classes_total // self.num_tasks
        print(f"Total number of classes: {self.num_classes_total} for dataset: {self.args.dataset}, num_tasks: {self.num_tasks}")
        self.batch_size = args.batch_size
        if "imagenet" in self.args.dataset:
            self.batch_size = 16
        self.n_client_epochs = args.n_client_epochs
        self.multi_head_flag = args.multi_head_flag
        self.L = args.L
        self.eps = args.eps
        self.optim_name = str(args.optim).lower()
        
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
        self.gamma_flag = args.gamma_flag
        self.fedtrack_flag = args.fedtrack_flag
        if self.IAG_batch_flag == True:
            self.train_shuffle = False
        else:
            self.train_shuffle = True
        print(f"Training shuffle: {self.train_shuffle}")
        self.adaptive_flag = args.adaptive_flag
        self.adaptive_lr_batch_flag = args.adaptive_lr_batch_flag 
        self.adaptive_lr_step_flag = args.adaptive_lr_step_flag 
        self.adaptive_lr_round_flag = args.adaptive_lr_round_flag 
        assert self.IAG_batch_flag != self.train_shuffle, "IAG_batch_flag and train_shuffle should be of opposite bool values"
        self.add_f_adapt_inside = args.add_f_adapt_inside
        self.add_f_adapt = args.add_f_adapt
        if self.IAG_batch_flag:
            assert self.delayed_grad_batch_flag, "For IAG_batch, delayed_grad_batch_flag should be True"
        if not self.grad_f_flag:
            # assert not self.adaptive_flag, "For adaptive learning((adaptive_flag=True)), grad_f_flag should be True"
            print("For adaptive learning((adaptive_flag=True)), grad_f_flag should be True")
        if self.adaptive_lr_step_flag and self.adaptive_lr_round_flag:
            raise ValueError("Both the step and round adaptive learning flags can not be True")
        if not self.adaptive_flag:
            assert not self.proxy_adaptive_lr, "For no adaptive learning (adaptive_flag=False), proxy_adaptive_lr should be False"
        self.device = torch.device(
            f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
            )
    
    def _calc_gamma(self, beta, ai, alp, inn, ):
        """Calculate forgetting gamma for each client"""
        ai_2 = self._inner_prod_sum(ai, ai)
        first_term = (self.L*beta*ai_2) / (2*self.num_clients)
        second_term = - beta* (1- self.L*alp) * inn

        return first_term + second_term
    
    def _get_offset_loss(self, logits, targets, total_classes, old_classes, offset_old=True):
        

        mask = torch.zeros_like(logits)
        if offset_old:
            mask[:, : old_classes ] = float('-inf')
        else:
            mask[:, old_classes : ] = float('-inf')
        # Apply the mask to logits
        masked_logits = logits + mask
        # Compute cross-entropy loss with masked logits
        loss = F.cross_entropy(masked_logits, targets)

        try:
            del mask
            del offset
        except:
            pass
        # gc.collect()
        return loss

    
    
    def server(self, ):
        """Implementation of the server strategy"""
        if self.args.model_name == "MLP":
            self.global_model = MLP_inc(input_size=28*28, hidden_size=512, n_classes=10, num_tasks=self.num_tasks).to(self.device)
        elif self.args.model_name == "resnet18":
            self.global_model = ResNet18(num_classes_total=self.num_classes_total, num_tasks=self.num_tasks, args={"dataset":self.args.dataset}).to(self.device)
        elif self.args.model_name == "resnet50":
            self.global_model = ResNet50(num_classes_total=self.num_classes_total, num_tasks=self.num_tasks, args={"dataset":self.args.dataset}).to(self.device)
        else:
            raise NotImplementedError
        print(f"Training using model: {type(self.global_model)}")
        self.num_clients = max(int(self.args.frac * self.args.n_clients),1)
        self.client_buffer_size = [self.args.initial_buffer_size]*self.num_clients
        
        self.memory_buffer = [ParametricBuffer(max_size=self.client_buffer_size[0],groupby='task',selection_strategy=RandomExemplarsSelectionStrategy())
                                                    for _ in range(self.num_clients)
                                                    ]
        
        self.task_store = []
        self.global_gamma_dict = {}
        
        self.cl_models_list = [copy.deepcopy(self.global_model) for _ in range(self.num_clients)]
        # pass
        for task_id in range(self.num_tasks):
            self.task_id = task_id
            self.classes_seen = (self.num_classes_total//self.num_tasks)*(self.task_id+1)
            self.old_classes = (self.num_classes_total//self.num_tasks)*(self.task_id)
            exp_train = self.bm.train_stream[self.task_id]
            _ = self.mini_server(exp_train)
            print(f"End of task: {self.task_id}")#, server, global model param_sum: {sum(param.sum() for param in self.global_model.parameters())}")
            # increment output layers and re-initialize weights from old model
            print(f"Updating output layer at the end of task: {self.task_id}")
            if self.task_id < self.num_tasks - 1:
                with torch.no_grad():
                    self.global_model.update_task_classifiers(self.task_id)
                    self.global_model.to(self.device)
                    for cl_id in range(self.num_clients):
                        self.cl_models_list[cl_id].update_task_classifiers(self.task_id) # update the classifier layer
                        self.cl_models_list[cl_id].to(self.device)
                        self.cl_models_list[cl_id].load_state_dict(self.global_model.state_dict()) # Load global model weights
            print(f"Output layer updation completed at the end of task: {self.task_id}")
            after_update_metric = {task_id: self.test_with_task_id(task_id) for task_id in range(self.task_id + 1) }
            print(f"After updating output layer task metric {after_update_metric}")
            # test model
            test_loss, test_acc = self.test()
            print(f"From server After task: {self.task_id}: Test loss: {test_loss}, Test accuracy: {test_acc}")
            if self.gamma_flag:
                if self.task_id >0:
                    self.global_gamma_dict[self.task_id] = self.client_gamma
        
        print("Training completed")
        print(f"Task-wise test metric progression dict:")
        for entry in self.task_store:
            print(f"{entry},")
        # Get avg. accuracy ancd forgetting metric
        _ = get_acc_forgetting(self.task_store, num_tasks=self.num_tasks)
        
        with open(global_result_path+"final_result"+'.pkl', 'wb') as f:
            pickle.dump(self.task_store, f)
        
        if self.gamma_flag:
            with open(global_result_path+"final_gamma"+'.pkl', 'wb') as f:
                pickle.dump(self.global_gamma_dict, f)
        

    def create_optimizer(self, task_id, model):
        if self.optim_name.lower() == "sgd":
            if (task_id == 0):
                current_lr = self.args.lr # for proxy adaptive learning rate
                optim = torch.optim.SGD([
                                        {'params': [param for name, param in model.named_parameters() if name != 'temperature']},  # Base model parameters
                                        {'params': [model.temperature], 'lr': 0.01}  # Temperature learning rate
                                        ],
                                        lr=self.args.lr,
                                        momentum=self.args.momentum,
                                        weight_decay=5e-4,
                                        )
                return optim
            else :
                current_lr = self.args.lr_retrain # for proxy adaptive learning rate
                optim = torch.optim.SGD([
                                        {'params': [param for name, param in model.named_parameters() if name != 'temperature']},  # Base model parameters
                                        {'params': [model.temperature], 'lr': 0.01}  # Temperature learning rate
                                        ],
                                        lr=self.args.lr_retrain,
                                        momentum=self.args.momentum,
                                        weight_decay=5e-4,
                                        )
                return optim
        elif self.optim_name.lower() == "adam":
            if (self.task_id == 0):
                current_lr = self.args.lr # for proxy adaptive learning rate
                optim = torch.optim.Adam([
                                        {'params': [param for name, param in model.named_parameters() if name != 'temperature']},  # Base model parameters
                                        {'params': [model.temperature], 'lr': 0.01}  # Temperature learning rate
                                        ],
                                        lr=self.args.lr,
                                        )
                return optim
            else :
                current_lr = self.args.lr_retrain # for proxy adaptive learning rate
                optim = torch.optim.Adam([
                                        {'params': [param for name, param in model.named_parameters() if name != 'temperature']},  # Base model parameters
                                        {'params': [model.temperature], 'lr': 0.01}  # Temperature learning rate
                                        ],
                                        lr=self.args.lr_retrain,
                                        )
                return optim
    
    def _train_on_memory(self, cl_id, exp_train,):
        pass


    def mini_server(self, exp_train):
        print(f"(PRINT) Task id {self.task_id}, training on {exp_train.classes_in_this_experience}")
        cl_data_indices = sampler(exp_train.dataset, n_clients=self.num_clients, n_classes=exp_train.classes_in_this_experience, alpha=self.args.alpha)

        save_train_losses = []
        save_train_accs = []
        save_test_losses = []
        save_test_accs = []
        train_losses, train_accs = [], []
        test_losses,test_accs = [], []

        client_datasets = []
        for cl_id in range(self.num_clients):
            sub_exp_train = copy.deepcopy(exp_train)
            sub_exp_train.dataset = sub_exp_train.dataset.subset(cl_data_indices[cl_id])
            client_datasets.append(sub_exp_train)
        self.client_gamma = {i: [] for i in range(self.num_clients)}
        
        for epoch in tqdm(range(self.args.n_rounds),):
            self.global_epoch = epoch
            clients_losses, clients_accs = [], []
            clients_test_losses, clients_test_accs = [], []
            idx_clients = [i for i in range(self.num_clients)]
            
            if self.IAG_flag or self.proxy_adaptive_lr:
                iag_grads_client_list = [self._calc_iag_grads(cl_idx, client_datasets[cl_idx]) for cl_idx in range(self.num_clients)]
                grads_iag_dict_server = self._process_iag_grads(iag_grads_client_list,) #--check for grad divison 
            
            else:
                iag_grads_client_list = [None]*self.num_clients
                grads_iag_dict_server = None
            
            if (self.grad_f_flag) and (self.task_id > 0):
                if self.kick_memory and self.IAG_flag: 
                    grads_g_full = self._get_full_grads_g(grads_iag_dict_server)
                    grads_f_clients_list_samples = [self._get_memory_grad_inc(cl_idx, grads_g= grads_g_full) for cl_idx in range(self.num_clients)]
                    grads_f_clients_list = [cl[0] for cl in grads_f_clients_list_samples]
                    elig_clients = sum([1 if cl[1]>0 else 0 for cl in grads_f_clients_list_samples])
                else:
                    grads_f_clients_list_samples = [self._get_memory_grad_inc(cl_idx,) for cl_idx in range(self.num_clients)]
                    grads_f_clients_list = [cl[0] for cl in grads_f_clients_list_samples]
                    elig_clients = sum([1 if cl[1]>0 else 0 for cl in grads_f_clients_list_samples])
                grads_f_server_list = [sum(grads) for grads in zip(*grads_f_clients_list)]
                # Average them out
                grads_f_server_list = [torch.div(grad, max(1,elig_clients)) for grad in grads_f_server_list]
            else:
                grads_f_server_list, grads_f_server_dict = None, None
            
            # Get proxy LR
            if (self.proxy_adaptive_lr) and (self.task_id > 0):
                    current_lr = torch.tensor(self.args.lr_retrain)
                    self.alp_proxy, self.bet_proxy = self._get_proxy_adaptive_lr(grads_f_server_list, grads_iag_dict_server, current_lr)
            elif self.task_id > 0:
                current_lr = torch.tensor(self.args.lr_retrain)
                self.alp_proxy, self.bet_proxy = torch.tensor(self.args.lr_retrain), torch.tensor(self.args.lr_retrain)
            else:
                current_lr = torch.tensor(self.args.lr)
                self.alp_proxy, self.bet_proxy = torch.tensor(self.args.lr), torch.tensor(self.args.lr)
            print(f"Inside mini_server server level proxy adaptive learning rate: alp: {self.alp_proxy}, beta: {self.bet_proxy}, current_lr: {current_lr}")
            
            for cl_id in idx_clients:
                
                before_task_metric = {task_id: self.test_with_task_id(task_id, model=self.cl_models_list[cl_id]) for task_id in range(self.task_id + 1) }
                print(f"Before task metric for client {cl_id}: ", before_task_metric)
                cl_loss, cl_acc = self._train_client_adap_inc(cl_id, client_datasets[cl_id], grads_iag_dict_server, iag_grads_client_list[cl_id], grads_f_server_list)
                
                clients_losses.append(cl_loss)
                clients_accs.append(cl_acc)
                cl_test_loss, cl_test_acc = self.test(model=self.cl_models_list[cl_id])
                clients_test_losses.append(cl_test_loss)
                clients_test_accs.append(cl_test_acc)
                print(f"From mini-server Task: {self.task_id}, client: {cl_id}, Server round: {epoch} training loss: {cl_loss}, training accuracy: {cl_acc}")
                print(f"From mini-server Task: {self.task_id}, client: {cl_id}, Server round: {epoch} test loss: {cl_test_loss}, test accuracy: {cl_test_acc}")
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
        
        print(f"At the end of task {self.task_id} length of memory buffers: {[(client_idx, self.client_buffer_size[client_idx]) for client_idx in range(self.num_clients)]}")
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

        # Get the test accuracies
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
        grads_list: A list of tuples containing the name of the parameter and the gradient. If batched, one per batch one per batch
        """
        if model == None:
            model = copy.deepcopy(self.cl_models_list[client_idx])
        else:
            model = copy.deepcopy(model)
        
        optimizer = self.create_optimizer(self.task_id, model)
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        
        train_loader, _ = self._get_dataloader(sub_exp_train.dataset, only_train=True, shuffle=self.train_shuffle)
        task_id_device = torch.tensor(copy.copy(self.task_id)).to(self.device)
        
        if self.IAG_batch_flag:
            model.train()
            grads_dict = {}
            for idx, (data, target, task_id_label) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                logits = model(data)
                # loss = criterion(logits, target)
                loss = self._get_offset_loss(logits, target, self.classes_seen, self.old_classes, offset_old=True)
                # loss = criterion(logits, target)
                loss.backward()
                with torch.no_grad():
                    grads_batch = {str(name): param.grad.clone().detach() if param.grad is not None else torch.zeros_like(param) for (name, param) in model.named_parameters()}

                optimizer.zero_grad()
                grads_dict[str(idx)] = grads_batch
        else:
            model.train()
            grads_dict = {}
            for idx, (data, target, task_id_label) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                logits = model(data)
                # loss = criterion(logits, target)
                loss = self._get_offset_loss(logits, target, self.classes_seen, self.old_classes, offset_old=True)
                # loss = criterion(logits, target)
                loss.backward()
                
            with torch.no_grad():
                grads_full = {str(name): torch.div(param.grad.clone().detach(),len(train_loader)) if param.grad is not None else torch.zeros_like(param) for (name, param) in model.named_parameters()}
            
            optimizer.zero_grad()
            grads_dict["full"] = grads_full
        
        try:
            del model, optimizer, criterion, train_loader, task_id_device, data, target, logits, loss
        except:
            # print("Error in deleting variables inside calc_iag_grads")
            pass
        # gc.collect()
        
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
        
        # gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        return grads_g

    def _get_proxy_adaptive_lr(self, grads_f_server_dict, grads_iag_dict_server, current_lr,):
        """Get the adaptive learning rate for the server"""
        # grads_f = [grad for name, grad in grads_f_server_dict.items()]
        grads_f = grads_f_server_dict # already a list
        if grads_f_server_dict is not None:
            grads_g = self._get_full_grads_g(grads_iag_dict_server)
        else:
            raise ValueError("grads_f_server_dict is None")
        
        inn = self._inner_prod_sum(grads_f, grads_g)
        norm_f_2 = self._inner_prod_sum(grads_f, grads_f)
        norm_g_2 = self._inner_prod_sum(grads_g, grads_g)
        # print(f"Inner product for proxy is: {inn} for norm_f {norm_f_2}, norm_g {norm_g_2}")
        if inn <= 0:
            alp_proxy = current_lr*(1- torch.div(inn,max(norm_f_2, self.eps)))
            beta_proxy = current_lr
        else:
            alp_proxy = current_lr
            inn = torch.clamp(inn, min=self.eps,)
            beta_proxy = ((1- current_lr*self.L) / max((self.L * norm_g_2), self.eps))* inn

        try:
            del grads_f, grads_g, inn, norm_f_2, norm_g_2
        except:
            pass
        # gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        return alp_proxy, beta_proxy

    def _process_iag_grads(self, iag_grads_dict_list:list,):
        
        if self.IAG_batch_flag:
            all_batch_ids = set().union(*iag_grads_dict_list)
            batch_dicts = {batch_id: [client_dict[batch_id] for client_dict in iag_grads_dict_list if batch_id in client_dict.keys()] for batch_id in all_batch_ids }
            param_keys_init = iag_grads_dict_list[0][str(0)].keys() #model param names
            grads_iag_dict =  {batch_id: {k: sum(d[k] for d in batch_dicts[batch_id]) for k in param_keys_init} for batch_id in batch_dicts.keys()}
            grads_iag_dict = {batch_id: {k: torch.div(v, len(batch_dicts[batch_id])) for k, v in grads_iag_dict[batch_id].items()} for batch_id in grads_iag_dict.keys()}
            
            del all_batch_ids, batch_dicts, param_keys_init, 
        else:
            iag_grads_dict_list = [d["full"] for d in iag_grads_dict_list if "full" in d.keys()]
            grads_iag_dict = {k: sum(d[k] for d in iag_grads_dict_list) for k in iag_grads_dict_list[0]}
            grads_iag_dict = {"full": {k : torch.div(v, len(iag_grads_dict_list)) for k, v in grads_iag_dict.items()}}

            del iag_grads_dict_list
        # gc.collect()
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
    
    def _get_memory_grads_new(self, client_idx, grads_g:list=None, model=None):
        """ Compute memory grads for each client - batched/full - currently only full verion is implemented"""
        if model is not None:
            model = copy.deepcopy(model)
        else:
            model = copy.deepcopy(self.cl_models_list[client_idx])
        optimizer = self.create_optimizer(self.task_id, model)
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        
        # class balanced in class-incremental
        storage_p = self.memory_buffer[client_idx]
        mem_batch_size = 16
        num_samples = 0
        sample_size = min(self.memory_sample_size * self.task_id, len(storage_p.buffer)) 
        sample_idx = random.sample(range(len(storage_p.buffer)), sample_size)
        memory_data = storage_p.buffer.subset(sample_idx)
        print(f"class-balanced memory buffer size for client {client_idx}: {len(storage_p.buffer)}, sampled size: {sample_size}")
        
        train_loader_memory, _ = self._get_dataloader(memory_data, batch_size=mem_batch_size, only_train=True) 
        model.train()
        for idx, (data, target, task_id_label) in enumerate(train_loader_memory):
            data, target = data.to(self.device), target.to(self.device),
            logits = model(data)
            loss = criterion(logits, target)
            # loss = self._get_offset_loss(logits, target, self.classes_seen, self.old_classes, offset_old=False)
            
            if grads_g is not None:
                grads_f = torch.autograd.grad(loss, model.parameters(), create_graph=False, allow_unused=True, retain_graph=True)
                with torch.no_grad():
                    # None is fine
                    grads_f = [grads_f[i].clone().detach() if grads_f[i] is not None else torch.zeros_like(param) for i, (name,param) in enumerate(model.named_parameters())]
                    inn = self._inner_prod_sum(grads_f, grads_g)
                if inn >= self.eps:
                    loss.backward() # accumulate memory data grad
                    num_samples += data.shape[0] #mem_batch_size
                else:
                    continue # remove that memory data
            else: # case for memory_data kick False
                loss.backward()
                num_samples += data.shape[0] #mem_batch_size
        if num_samples > 0: # Either selected samples or all samples
            grads_f_client = {str(name): torch.div(param.grad.clone().detach(), num_samples) if param.grad is not None else torch.zeros_like(param) for (name, param) in model.named_parameters()}
        else: # No samples taken
            grads_f_client = {str(name): torch.zeros_like(param) for (name, param) in model.named_parameters()}
        optimizer.zero_grad()

        print(f"Client:{client_idx} sampled {num_samples} from memory for kick_memory {self.kick_memory}")
        try:
            del model, optimizer, criterion, train_loader_memory, data, target, logits, loss
            del sample_size, sample_idx, buf_id
        except:
            pass
        # gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        return (grads_f_client, num_samples)

    
    
    def _inner_prod_sum(self, grad_list_1, grad_list_2):
        """Calculate inner product"""
        inn = torch.tensor(0.0).to(self.device)
        for grad1, grad2 in zip(grad_list_1, grad_list_2):
            product = torch.div(torch.dot(grad1.view(-1), grad2.view(-1)), len(grad1.view(-1)))
            inn += product
        del product
        # gc.collect()
        return inn
    
    def calc_new_grads_adaptive_batch(self, grads_f:list, grads_g:list, current_lr, alpha_lr_proxy=None, beta_lr_proxy=None,client_idx=None):
        """For Every Step Adaptive Learning
        Returns:
        new_grads: list of new gradients
        """
        if alpha_lr_proxy is None:
            alpha_lr_proxy = self.alp_proxy
        if beta_lr_proxy is None:
            beta_lr_proxy = self.bet_proxy
        alp_t, beta_t = alpha_lr_proxy, beta_lr_proxy
        
        if client_idx:
            
            grads_f = self.client_mem_grad

        inn = self._inner_prod_sum(grads_f, grads_g)
        norm_f_2 = self._inner_prod_sum(grads_f, grads_f)
        norm_g_2 = self._inner_prod_sum(grads_g, grads_g)
        
        
        if inn >0: 
            inn = torch.clamp(inn, min=self.eps,)
        
        beta = ((1-alp_t*self.L) / max(self.L * norm_g_2, self.eps)) * inn
        alpha = torch.mul(alp_t, (1- torch.div(inn,max(norm_f_2, self.eps))))
        # clamp values
        if self.optim_name == "adam":
            beta = torch.clamp(beta, max=0.001, min = 0.000001)
        elif self.optim_name == "sgd":
            beta = torch.clamp(beta, max=0.95, min= 0.0001)
        
        
        # divide by learning_rate for further use by optimizer
        mul_factor_1 = beta/max(current_lr,self.eps)
        mul_factor_2 = beta_t/max(current_lr,self.eps)
        new_grads_g = [torch.where(inn >0, mul_factor_1*grads_g[i], mul_factor_2*grads_g[i]) for i in range(len(grads_g))]

        # #Added 10 Oct'24
        if self.add_f_adapt_inside:
            new_grads_f = [torch.where(inn <=0, (alpha/current_lr)*grads_f[i], (alp_t/current_lr)*grads_f[i]) for i in range(len(grads_f))]
            new_grads_g = [f+g for (f,g) in zip(new_grads_f, new_grads_g)]

        try:
            del inn, norm_f_2, norm_g_2, mul_factor_1, mul_factor_2, beta
        except:
            pass
        # gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        return new_grads_g
    
    def calc_new_grads_adaptive_step(self, grads_f, grads_g, current_lr,alpha_lr_proxy=None, beta_lr_proxy=None, add_memory=False,client_idx=None):
        
        if alpha_lr_proxy is None:
            alpha_lr_proxy = self.alp_proxy
        if beta_lr_proxy is None:
            beta_lr_proxy = self.bet_proxy
        alp_t, beta_t = alpha_lr_proxy, beta_lr_proxy
        
        # grads_f, _ = self._get_memory_grad_inc(client_idx, )
        grads_f = self.client_mem_grad
        inn = self._inner_prod_sum(grads_f, grads_g)
        norm_g_2 = self._inner_prod_sum(grads_g, grads_g)
        norm_f_2 = self._inner_prod_sum(grads_f, grads_f)
        
        if inn >0: 
            inn = torch.clamp(inn, min=self.eps,)
       
        beta_t_i = (inn* (1- alp_t*self.L) / max(self.L * norm_g_2, self.eps) )
        if self.optim_name == "adam":
            beta_t_i = torch.clamp(beta_t_i, max=0.001, min = 0.000001)
        elif self.optim_name == "sgd":
            beta_t_i = torch.clamp(beta_t_i, max=0.95, min= 0.0001)

        alpha_t_i = torch.mul(alp_t, (1- torch.div(inn,max(norm_f_2, self.eps))))
        print("Client adaptive learning rate (step): ", alpha_t_i, beta_t_i)
        
         # divide by learning_rate for further use by optimizer
        mul_factor_1 = (beta_t_i/max(beta_t,self.eps))* (1/current_lr)
        mul_factor_2 = 1/max(current_lr,self.eps) # (1 = beta_t/beta_t)
        
        grads_g_new = [torch.where(inn >0, torch.mul(mul_factor_1, grads_g[i]), torch.mul(mul_factor_2, grads_g[i])) for i in range(len(grads_g))]

        # if add_memory:
        if self.add_f_adapt_inside:
            grads_f_new = [torch.where(inn <=0, torch.mul(alpha_t_i/current_lr, grads_f[i]), torch.mul(alp_t/current_lr, grads_f[i])) for i in range(len(grads_f))]
            new_grads = [f+g for (f,g) in zip(grads_f_new, grads_g_new)]
            
            del grads_f_new
        else:
            new_grads = grads_g_new
        
        try:
            del inn, beta_t_i, alpha_t_i, mul_factor_1, mul_factor_2
        except:
            pass
        # gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        return new_grads

    def _grad_updator(self, grads_list, model):
        with torch.no_grad():
            for idx, (_, param) in enumerate(model.named_parameters()):
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                param.grad = (torch.clamp(grads_list[idx].detach(), min=self.min_clip, max=self.max_clip))
        pass

    

    def _get_memory_grad_inc(self, client_idx,):
        model = copy.deepcopy(self.cl_models_list[client_idx])
        storage_p_all = self.memory_buffer[client_idx]
        storage_p_list = [v for v in storage_p_all.buffer_groups.values()]
        sample_size  = self.memory_sample_size # put small value in utils or argpasrse
        batch_size = 16
        optimizer = self.create_optimizer(self.task_id, model)
        grad_f_all = []
        batch_counts = 0
        for buf_id, storage_p in enumerate(storage_p_list):
            sample_idx = random.sample(range(len(storage_p.buffer)), sample_size)
            # print(f"client_id: {client_idx}, buf_id: {buf_id}, sample_idx: {sample_idx[0:8]}")
            memory_data = storage_p.buffer.subset(sample_idx)
            model.train()

            train_loader_memory, _ = self._get_dataloader(memory_data, batch_size=batch_size, only_train=True)
            head_start, head_end = buf_id*self.class_per_task, (buf_id+1)*self.class_per_task
            # print(f"head_start: {head_start}, head_end: {head_end}")
            for data, target, task_id_label in train_loader_memory:
                data, target = data.to(self.device), target.to(self.device)
                logits = model(data)
                
                mask = torch.full_like(logits, float('-inf'),)
                mask[:, : self.old_classes ] = 0.0
                # Apply the mask to logits
                masked_logits = logits + mask
                # Compute cross-entropy loss with masked logits
                loss = F.cross_entropy(masked_logits, target,)
                loss.backward()
                batch_counts += 1

                # num_samples += data.shape[0]
            grad_f_dict = {str(name): torch.div(param.grad.clone().detach(), len(train_loader_memory)) if param.grad is not None else torch.zeros_like(param) for (name, param) in model.named_parameters()}
            grad_f = [grad for name, grad in grad_f_dict.items()]
            grad_f_all.append(grad_f)
            optimizer.zero_grad()
        grad_f_final = [sum(grads)/batch_counts for grads in zip(*grad_f_all)]
        return grad_f_final, batch_counts
    
    
    
    def _calc_grads_for_inc(self, client_idx, grads_f:list, grads_g:list, current_lr, model):
        
        grads_f = [grads_f[i].clone().detach() if grads_f[i] is not None else torch.zeros_like(param) for i, (name,param) in enumerate(model.named_parameters())]
        grads_g = [grads_g[i].clone().detach() if grads_g[i] is not None else torch.zeros_like(param) for i, (name,param) in enumerate(model.named_parameters())]
        inn = self._inner_prod_sum(grads_f, grads_g)
        norm_g_2 = self._inner_prod_sum(grads_g, grads_g)
        norm_f_2 = self._inner_prod_sum(grads_f, grads_f)

        alpha_t_i = torch.mul(1, (1- torch.div(inn,max(norm_f_2, self.eps))))
        beta_t_i = (inn* (1- current_lr*self.L)) /((max(self.L * norm_g_2, self.eps)*current_lr ))
        if inn >0: # To avoid very small beta due to small inner product value
            inn = torch.clamp(inn, min=self.eps,)
        if self.optim_name == "adam":
            beta_t_i = torch.clamp(beta_t_i, max=0.001, min = 0.000001)
        elif self.optim_name == "sgd":
            beta_t_i = torch.clamp(beta_t_i, max=0.95, min= 0.00001)
        
        new_f = [torch.where(inn <=0, torch.mul(alpha_t_i, grads_f[i]), torch.mul(1, grads_f[i])) for i in range(len(grads_f))]
        new_g = [torch.where(inn >0, torch.mul(beta_t_i, grads_g[i]), torch.mul(1, grads_g[i])) for i in range(len(grads_g))] 
        new_f_g = [f+g for f,g in zip(new_f, new_g)]
        return new_f_g

    def _train_client_curr_inc_new(self, client_idx, sub_exp_train, server_iag_dict, client_iag_dict, grads_f_dict, adaptive=False):
        model = self.cl_models_list[client_idx]
        if grads_f_dict is None: 
            adaptive = False
        train_loader, test_loader = self._get_dataloader(sub_exp_train.dataset, shuffle=True, drop_last=False)
        optimizer = self.create_optimizer(self.task_id, model)
        current_lr = optimizer.param_groups[0]['lr']
        for epoch in range(self.args.n_client_epochs):
            model.train()
            optimizer.zero_grad()
            last_model_parameters = {name: param.clone().detach() for name, param in model.named_parameters()}
            if self.task_id > 0:
                self.client_mem_grad, _ = self._get_memory_grad_inc(client_idx, )
            for batch_id, (data, target, task_id_label) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                logits_c = model(data)
                loss_c = self._get_offset_loss(logits_c, target, self.classes_seen, self.old_classes, offset_old=True)
                loss_c.backward()
                if (self.IAG_flag) and (self.IAG_batch_flag) and (self.delayed_grad_batch_flag): # case 3
                    
                    new_grads_iag = {name: param.grad.clone().detach() if param.grad is not None else torch.zeros_like(param) for (name, param) in model.named_parameters()}
                    new_grads_iag = [server_iag_dict[str(batch_id)][name]  - client_iag_dict[str(batch_id)][name] + new_grads_iag[name] for name, param in model.named_parameters()]
                    
                    if adaptive and self.adaptive_lr_batch_flag:
                        grads_f = grads_f_dict # already a list
                        current_lr = optimizer.param_groups[-1]['lr']
                        grads_g_final = self.calc_new_grads_adaptive_batch(grads_f, new_grads_iag,current_lr=current_lr,alpha_lr_proxy=self.alp_proxy, beta_lr_proxy=self.bet_proxy, client_idx=client_idx)
                        _ = self._grad_updator(grads_g_final, model)
                    else:
                        _ = self._grad_updator(new_grads_iag, model)
                    optimizer.step()
                    optimizer.zero_grad()
                elif (self.IAG_flag) and (not self.IAG_batch_flag) and (self.delayed_grad_batch_flag): 
                    # with torch.no_grad():
                    new_grads_iag = {name: param.grad.clone().detach() if param.grad is not None else torch.zeros_like(param) for (name, param) in model.named_parameters()}
                    new_grads_iag = [server_iag_dict["full"][name]  - client_iag_dict["full"][name] + new_grads_iag[name] for name, param in model.named_parameters()]

                    if adaptive and self.adaptive_lr_batch_flag:
                        grads_f = grads_f_dict 
                        current_lr = optimizer.param_groups[-1]['lr']
                        grads_g_final = self.calc_new_grads_adaptive_batch(grads_f, new_grads_iag,current_lr, self.alp_proxy,self.bet_proxy,client_idx=client_idx)
                        _ = self._grad_updator(grads_g_final, model)                        
                    else:
                        _ = self._grad_updator(new_grads_iag, model)
                    optimizer.step()
                    optimizer.zero_grad()
                elif (not self.IAG_flag) and (not self.IAG_batch_flag) and (self.delayed_grad_batch_flag): 
                    grads_g = {name: param.grad.clone().detach() if param.grad is not None else torch.zeros_like(param) for (name, param) in model.named_parameters()}
                    grads_g = [param for name,param in grads_g.items()]
                    pass
                    if adaptive and self.adaptive_lr_batch_flag:
                        
                        grads_f = grads_f_dict # already a list
                        current_lr = optimizer.param_groups[-1]['lr']
                        grads_g_final = self.calc_new_grads_adaptive_batch(grads_f, grads_g, current_lr, self.alp_proxy, self.bet_proxy, client_idx=client_idx)
                        _ = self._grad_updator(grads_g_final, model)                            
                    else:
                        _ = self._grad_updator(grads_g, model)
                    optimizer.step()
                    optimizer.zero_grad()

                else: 
                    continue
            
            if (self.IAG_flag) and (not self.IAG_batch_flag) and (not self.delayed_grad_batch_flag): #case 1
                
                new_grads_iag = {name: param.grad.clone().detach() if param.grad is not None else torch.zeros_like(param) for (name, param) in model.named_parameters()}
                new_grads_iag = [server_iag_dict["full"][name]  - client_iag_dict["full"][name] + new_grads_iag[name] for name, param in model.named_parameters()]

                if adaptive and self.adaptive_lr_step_flag:
                    grads_f = grads_f_dict 
                    current_lr = optimizer.param_groups[-1]['lr']
                    grads_g_final = self.calc_new_grads_adaptive_step(grads_f, new_grads_iag, current_lr, self.alp_proxy, self.bet_proxy, add_memory=True,)
                    _ = self._grad_updator(grads_g_final, model)
                else:
                    _ = self._grad_updator(new_grads_iag, model)
                optimizer.step()
                optimizer.zero_grad()
            
            elif (self.adaptive_lr_step_flag) and (adaptive): 
                a_i = [last_model_parameters[name] - param.clone().detach() for (name, param) in model.named_parameters()] 
                assert grads_f_dict is not None, "grads_f is None in the cflag proposed training"
                # grads_f = [grad for name, grad in grads_f_dict.items()]
                grads_f = grads_f_dict # already a list
                # with torch.no_grad():
                current_lr = optimizer.param_groups[-1]['lr']            
                new_grads_g = self.calc_new_grads_adaptive_step(grads_f, a_i, current_lr=current_lr, alpha_lr_proxy=self.alp_proxy, beta_lr_proxy=self.bet_proxy, add_memory=True, client_idx=client_idx)
                _ = self._grad_updator(new_grads_g, model)
                               
                optimizer.step()
                optimizer.zero_grad()
            # Scheduler step
            # add memory for class incremental
            if self.task_id >0:
                pass
                # scheduler.step()
            # Metrics - Check training and test accuracy with local model E steps- at each local step

            task_metric_dict = {task_id: self.test_with_task_id(task_id, model=model) for task_id in range(self.task_id+1) }
            formatted_results = format_dict_with_precision(task_metric_dict)
            print(f"Client:{client_idx}, Epoch:{epoch}, Task:{self.task_id}, task_metric_dict: {task_metric_dict}")
        #save results
        with open(client_result_path+'test_accuracy_model'+str(client_idx)+'task'+str(self.task_id)+'.pkl', 'wb') as f:
             pickle.dump(task_metric_dict, f)
        print(f"--------"*20)

        try:
            del task_id_device, optimizer, criterion, train_loader, test_loader, last_model_parameters
        except:
            pass
        # gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

    def _train_client_curr_inc(self, client_idx, sub_exp_train, server_iag_dict, client_iag_dict, adaptive=False):
        model = self.cl_models_list[client_idx]
        train_loader, test_loader = self._get_dataloader(sub_exp_train.dataset, shuffle=True, drop_last=False)
        optimizer = self.create_optimizer(self.task_id, model)
        current_lr = optimizer.param_groups[0]['lr']
        for epoch in range(self.args.n_client_epochs):
            model.train()
            optimizer.zero_grad()
            last_model_parameters = {name: param.clone().detach() for name, param in model.named_parameters()}
            for batch_id, (data, target, task_id_label) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                logits_c = model(data)
                # loss = criterion(logits, target)
                loss_c = self._get_offset_loss(logits_c, target, self.classes_seen, self.old_classes, offset_old=True)
                grads_g = torch.autograd.grad(loss_c, model.parameters(), create_graph=False, allow_unused=True, retain_graph=True)
                grads_g = [grads_g[i].clone().detach() if grads_g[i] is not None else torch.zeros_like(param) for i, (name,param) in enumerate(model.named_parameters())]
                if self.IAG_flag:
                    server_iag = [grad for name,grad in server_iag_dict["full"].items()]
                    client_iag = [grad for name,grad in client_iag_dict["full"].items()]
                    grads_g = [server_iag[idx]  - client_iag[idx] + grads_g[idx] for idx in range(len(grads_g))]

                if adaptive and self.adaptive_lr_batch_flag:
                    grads_f, _ = self._get_memory_grad_inc(client_idx, self.memory_sample_size)
                    grads_g = self.calc_new_grads_adaptive_batch(grads_f, grads_g, current_lr)
                    final_grads = self._calc_grads_for_inc(client_idx, grads_f, grads_g, current_lr, model)
                else:
                    final_grads = [grads_g[i].clone().detach() if grads_g[i] is not None else torch.zeros_like(param) for i, (name,param) in enumerate(model.named_parameters())]
                _ = self._grad_updator(final_grads, model)
                optimizer.step()
                optimizer.zero_grad()
            
            if adaptive and self.adaptive_lr_step_flag:
                grads_f = self._get_memory_grad_inc(client_idx, self.memory_sample_size)
                if self.IAG_flag:
                    # a_i exists
                    final_grads = self._calc_grads_for_inc(client_idx, grads_f, a_i, current_lr, model)
                    _ = self._grad_updator(final_grads, model)
                else:
                    a_i = [last_model_parameters[name] - param.clone().detach() for (name, param) in model.named_parameters()] # cumulative grads throug params
                    final_grads = self._calc_grads_for_inc(client_idx, grads_f, a_i, current_lr, model)
                _ = self._grad_updator(final_grads, model)
            elif adaptive:
                pass # add grad_updator
                #train on memory
                grads_f = self._get_memory_grad_inc(client_idx, self.memory_sample_size)
                _ = self._grad_updator(grads_f, model)
            optimizer.step()
            optimizer.zero_grad()

        return
    
    def _train_client_adap_inc(self, client_idx, sub_exp_train, server_iag_dict, client_iag_dict, grads_f_server):# adaptive:bool=False):
        if self.task_id == 0 or not self.adaptive_flag:
            _ = self._train_client_curr_inc_new(client_idx, sub_exp_train,server_iag_dict, client_iag_dict, grads_f_dict=None, adaptive=False)
        else:
            _ = self._train_client_curr_inc_new(client_idx, sub_exp_train,server_iag_dict, client_iag_dict,grads_f_dict=grads_f_server, adaptive=self.adaptive_flag)

            model = self.cl_models_list[client_idx] 
            model.train()
            optimizer = self.create_optimizer(self.task_id, model)
            
            if self.kick_memory: # not doing
                grads_f_new, num_samples = self._get_memory_grad_inc(client_idx,)
            else:
                grads_f_new, num_samples = self._get_memory_grad_inc(client_idx,)
            
            # with torch.no_grad():
            if self.add_f_adapt:
                pass
                _ = self._grad_updator(grads_f_new, model)
                optimizer.step() # if commneted then no grad f part
            optimizer.zero_grad()

            del optimizer
            if self.gamma_flag:
                # get the gamma and grad f for plots
                global_model_parameters = {name: param.clone().detach() for name, param in self.global_model.named_parameters()}
                ai = [global_model_parameters[name] - param_client.clone().detach() for (name, param_client) in model.named_parameters()] # cumulative grads
                inn = self._inner_prod_sum(grads_f_new, ai)
                gamma_i = self._calc_gamma(self.bet_proxy, ai, self.alp_proxy, inn)
                # print(f"Gamma_i: {gamma_i}")
                self.client_gamma[client_idx].append(gamma_i.item())
        
        ## Fetch scores
        train_loader, _ = self._get_dataloader(sub_exp_train.dataset, only_train=True)
        train_loss, train_acc = self.test(train_loader, model=self.cl_models_list[client_idx])
        
        # train_task_metric_dict = {task_id: self.train_metric_with_task_id(task_id, model=self.cl_models_list[client_idx]) for task_id in range(self.task_id+1) }
        task_metric_dict = {task_id: self.test_with_task_id(task_id, model=self.cl_models_list[client_idx]) for task_id in range(self.task_id+1) }
        formatted_results = format_dict_with_precision(task_metric_dict)
        print(f"At the end of Task: {self.task_id}, client: {client_idx}, Server epoch: {self.global_epoch}, TEST_stats (f+g) on local model: {formatted_results}")
        
        try:
            del task_metric_dict
        except:
            pass
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        return train_loss, train_acc
    ############### Incremental part ends ############################

    def _get_new_fplus_g_del(self, grads_f:list, grads_g:list, current_lr, model=None):
        alp_t, beta_t = self.alp_proxy, self.bet_proxy
        grads_f = [grads_f[i].clone().detach() if grads_f[i] is not None else torch.zeros_like(param) for i, (name,param) in enumerate(model.named_parameters())]
        grads_g = [grads_g[i].clone().detach() if grads_g[i] is not None else torch.zeros_like(param) for i, (name,param) in enumerate(model.named_parameters())]
        inn = self._inner_prod_sum(grads_f, grads_g)
        norm_g_2 = self._inner_prod_sum(grads_g, grads_g)
        norm_f_2 = self._inner_prod_sum(grads_f, grads_f)

        alpha_t_i = torch.mul(alp_t, (1- torch.div(inn,max(norm_f_2, self.eps))))
        beta_t_i = inn* (1- current_lr*self.L) / (max(self.L * norm_g_2, self.eps)*current_lr )
        if inn >0: # To avoid very small beta due to small inner product value
            inn = torch.clamp(inn, min=self.eps,)
        if self.optim_name == "adam":
            beta_t_i = torch.clamp(beta_t_i, max=0.001, min = 0.000001)
        elif self.optim_name == "sgd":
            beta_t_i = torch.clamp(beta_t_i, max=0.95, min= 0.0001)
        
        mul_factor_1 = (beta_t_i/max(beta_t,self.eps))* (1/current_lr)
        mul_factor_2 = 1/max(current_lr,self.eps) # (1 = beta_t/beta_t)

        alp_mul_factor_1 = torch.div(alpha_t_i, alp_t)
        alp_mul_factor_2 = torch.div(1, current_lr)
        
        new_f = [torch.where(inn <=0, torch.mul(alp_mul_factor_1, grads_f[i]), torch.mul(alp_mul_factor_2, grads_f[i])) for i in range(len(grads_f))]
        new_g = [torch.where(inn >0, torch.mul(mul_factor_1, grads_g[i]), torch.mul(mul_factor_2, grads_g[i])) for i in range(len(grads_g))] 
        new_f_g = [f+g for f,g in zip(new_f, new_g)]
        return new_f_g        

    def _train_client_curr_inc_add_f(self, client_idx, sub_exp_train, server_iag_dict:dict, client_iag_dict:dict, grads_f_dict:dict=None, adaptive:bool=False, add_f:bool=False):
        pass

    def _train_client_curr(self, client_idx, sub_exp_train, server_iag_dict:dict, client_iag_dict:dict, grads_f_dict:dict=None, adaptive:bool=False): 
        """Train a client on current task data
        Only handles the current training part - on g
        """
        task_id_device = torch.tensor(copy.copy(self.task_id)).to(self.device)
        model = self.cl_models_list[client_idx] 
        if grads_f_dict is None: 
            adaptive = False
       
        # model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = self.create_optimizer(self.task_id, model)
        lr_lmbda = lambda epoch: 1/math.sqrt(self.args.n_client_epochs)
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lmbda, last_epoch=-1)
        # Get the train loader and test loader
        train_loader, test_loader = self._get_dataloader(sub_exp_train.dataset, shuffle=self.train_shuffle, drop_last=False)
        optimizer.zero_grad()
        for epoch in range(self.args.n_client_epochs):
            model.train()
            last_model_parameters = {name: param.clone().detach() for name, param in model.named_parameters()}
            for batch_id, (data, target, task_id_label) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                logits = model(data)
                # loss = criterion(logits, target)
                loss = self._get_offset_loss(logits, target, self.classes_seen, self.old_classes, offset_old=True)
                loss.backward()
                
                
                if (self.IAG_flag) and (self.IAG_batch_flag) and (self.delayed_grad_batch_flag): # case 3
                    # print("case 3")
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
                
                elif (not self.IAG_flag) and (not self.IAG_batch_flag) and (self.delayed_grad_batch_flag): 
                    with torch.no_grad():
                        grads_g = {name: param.grad.clone().detach() if param.grad is not None else torch.zeros_like(param) for (name, param) in model.named_parameters()}
                        grads_g = [param for name,param in grads_g.items()]
                        
                        if adaptive and self.adaptive_lr_batch_flag:
                            # print("case NCCL, check train_adap outer loop adding grads_f also")
                            grads_f = [grad for name, grad in grads_f_dict.items()]
                            current_lr = optimizer.param_groups[-1]['lr']
                            grads_g_final = self.calc_new_grads_adaptive_batch(grads_f, grads_g, current_lr, self.alp_proxy, self.bet_proxy)
                            _ = self._grad_updator(grads_g_final, model)                            
                        else:
                            _ = self._grad_updator(grads_g, model)
                    
                    optimizer.step()
                    optimizer.zero_grad()

                else: # partial for case 1
                    #accumulate grads and then update
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
            
            elif (self.adaptive_lr_step_flag) and (adaptive): # Case: adaptive with any type of batched IAG, (per local epoch adaptive), this part only does adaptation
                # print("case adaptive with any type of batched IAG or non IAG, (only Full IAG not considered) (per local epoch adaptive)")
                a_i = [last_model_parameters[name] - param.clone().detach() for (name, param) in model.named_parameters()] # cumulative grads throug params
                assert grads_f_dict is not None, "grads_f is None in the cflag proposed training"
                grads_f = [grad for name, grad in grads_f_dict.items()]
                with torch.no_grad():
                    current_lr = optimizer.param_groups[-1]['lr']            
                    new_grads_g = self.calc_new_grads_adaptive_step(grads_f, a_i, current_lr=current_lr, alpha_lr_proxy=self.alp_proxy, beta_lr_proxy=self.bet_proxy, add_memory=False)
                    _ = self._grad_updator(new_grads_g, model)
                               
                optimizer.step()
                optimizer.zero_grad()
            
            
            if self.task_id >0:
                pass
                

            task_metric_dict = {task_id: self.test_with_task_id(task_id, model=model) for task_id in range(self.task_id+1) }
            formatted_results = format_dict_with_precision(task_metric_dict)
        #save results
        with open(client_result_path+'test_accuracy_model'+str(client_idx)+'task'+str(self.task_id)+'.pkl', 'wb') as f:
             pickle.dump(task_metric_dict, f)
        print(f"--------"*20)

        try:
            del task_id_device, optimizer, criterion, train_loader, test_loader, last_model_parameters
        except:
            pass
        # gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

    def _train_client_adap(self,
                client_idx,
                sub_exp_train, server_iag_dict:dict,
                client_iag_dict:dict, grads_f_server:dict=None,
                # adaptive:bool=False
                ):
        
        """Trains a client for one global server round on both the memory data and the current task data"""
        if (self.task_id == 0) or (not self.grad_f_flag):
            _ = self._train_client_curr(client_idx, sub_exp_train, server_iag_dict=server_iag_dict, client_iag_dict=client_iag_dict, grads_f_dict=None, adaptive=False)       
        elif (self.task_id > 0) and (grads_f_server is not None): 
            if self.adaptive_lr_round_flag:
                # Adapt at each server round only
                _ = self._train_client_curr(client_idx, sub_exp_train, server_iag_dict=server_iag_dict, client_iag_dict=client_iag_dict, grads_f_dict=None, adaptive=False)
            else:
                _ = self._train_client_curr(client_idx, sub_exp_train, server_iag_dict=server_iag_dict, client_iag_dict=client_iag_dict, grads_f_dict=grads_f_server, adaptive=self.adaptive_flag)
            
            model = self.cl_models_list[client_idx] 
            model.train()
            with torch.no_grad():
                global_model_parameters = {name: param.clone().detach() for name, param in self.global_model.named_parameters()}
                a_i = [global_model_parameters[name] - param_client.clone().detach() for (name, param_client) in self.cl_models_list[client_idx].named_parameters()] 
                assert grads_f_server is not None, "grads_f is None in the cflag proposed training"
                grads_f = [grad for name, grad in grads_f_server.items()]             
                # optimizer = self.opt_lists[client_idx]
                optimizer = self.create_optimizer(self.task_id, model)
                current_lr = optimizer.param_groups[-1]['lr']
                if self.adaptive_lr_round_flag: 
                    new_grads = self.calc_new_grads_adaptive_step(grads_f, a_i, current_lr=current_lr, alpha_lr_proxy=self.alp_proxy, beta_lr_proxy=self.bet_proxy, add_memory=True)
                    # Already clamped inside calc_new_grads_adaptive_step
                    _ = self._grad_updator(new_grads, model)
                else:
                    pass 
                    
                    
                    new_grads = [f + g for (f,g) in zip(grads_f, a_i)]
                    _ = self._grad_updator(new_grads, model)
                try:
                    del a_i, grads_f, global_model_parameters
                except:
                    pass
                    print("not fedtrack")
                optimizer.step()
                optimizer.zero_grad()
            

        ## Fetch scores
        train_loader, _ = self._get_dataloader(sub_exp_train.dataset, only_train=True)
        train_loss, train_acc = self.test(train_loader, model=self.cl_models_list[client_idx])

        task_metric_dict = {task_id: self.test_with_task_id(task_id, model=self.cl_models_list[client_idx]) for task_id in range(self.task_id+1) }
        formatted_results = format_dict_with_precision(task_metric_dict)
        print(f"(E local, 1 memory) Task: {self.task_id}, client: {client_idx}, Server epoch: {self.global_epoch}, TEST_stats (f+g) on local model: {formatted_results}")
        
        

        try:
            del train_task_metric_dict, task_metric_dict, global_model_parameters, a_i, grads_f, current_lr, new_grads, optimizer
        except:
            pass
        # gc.collect()
        return train_loss, train_acc
    
    def _train_client_adap_new_not_using(self,
                client_idx,
                sub_exp_train, server_iag_dict:dict,
                client_iag_dict:dict, grads_f_server:dict=None,
                
                ):
        
        """Trains a client for one global server round on both the memory data and the current task data"""
        if (self.task_id == 0) or (not self.grad_f_flag):
            _ = self._train_client_curr(client_idx, sub_exp_train, server_iag_dict=server_iag_dict, client_iag_dict=client_iag_dict, grads_f_dict=None, adaptive=False)
        elif (self.task_id > 0) and (grads_f_server is not None):
            
            _ = self._train_client_curr(client_idx, sub_exp_train, server_iag_dict=server_iag_dict, client_iag_dict=client_iag_dict, grads_f_dict=grads_f_server, adaptive=self.adaptive_flag)
            

            ### Commenting from here - 10 Oct'24
            model = copy.deepcopy(self.global_model)
            model.train()
            optimizer = self.create_optimizer(self.task_id, model)
            
            grads_g_cl = [self._calc_iag_grads(client_idx, sub_exp_train)]
            grads_g_dict = self._process_iag_grads(grads_g_cl)
            grads_g = self._get_full_grads_g(grads_g_dict)
            # Take memory grads
            if self.kick_memory:
                grads_f_dict, _ = self._get_memory_grads_new_para_buf(client_idx,grads_g=grads_g, model=model)
            else:
                grads_f_dict, _ = self._get_memory_grads_new_para_buf(client_idx,grads_g=None, model=model)
            grads_f_new = [param for name, param in grads_f_dict.items()]

            with torch.no_grad():
                _ = self._grad_updator(grads_f_new, model)
            optimizer.step()
            optimizer.zero_grad()
            del optimizer
            if self.gamma_flag:
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
        print(f"(E local, 1 memory) Task: {self.task_id}, client: {client_idx}, Server epoch: {self.global_epoch}, TEST_stats (f+g) on local model: {formatted_results}")
        

        try:
            del train_task_metric_dict, task_metric_dict
        except:
            pass
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        return train_loss, train_acc

    def memory_buffer_updator(self, client_idx, sub_exp_train, name=None ):
        """Function to update memory buffer"""
        
        strategy_state = SimpleNamespace(experience=sub_exp_train)
        storage_p = self.memory_buffer[client_idx]
        if self.task_id >0:
            self.client_buffer_size[client_idx] += self.args.initial_buffer_size
            storage_p.resize(strategy_state, self.client_buffer_size[client_idx])
        _ = storage_p.update(strategy_state)
        
    def _get_dataloader(self, training_dataset, batch_size=None, only_train=False, shuffle=None, drop_last=None):
        """Retuns dataloader"""
        if batch_size == None:
            batch_size = self.batch_size
        if shuffle is None:
            shuffle = True
        if drop_last is None:
            drop_last = False
        
        train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        if not only_train:
            test_set = self.bm.test_stream[self.task_id].dataset
            if "imagenet" in self.args.dataset:
                test_loader = DataLoader(test_set, batch_size=32)
            else:
                test_loader = DataLoader(test_set, batch_size=128)
        else:
            test_loader = None
        
        return train_loader, test_loader
    
    def test(self, test_loader=None, model=None, task_id=None, test_classes=None):
        """Test on the model. If no model passed as an argument then test on the server model."""
        del_model = False
        if model == None:
            model = copy.deepcopy(self.global_model)
            del_model = True
        model.eval()
        criterion = nn.CrossEntropyLoss()
        if "imagenet" in self.args.dataset:
            batch_size = 32
        else:
            batch_size = 128
        if test_loader == None:
            test_set = self.bm.test_stream[self.task_id].dataset
            test_loader = DataLoader(test_set, batch_size=batch_size)
        if task_id == None:
            task_id_device = torch.tensor(copy.copy(self.task_id)).to(self.device)
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for idx, (data, target, task_id_label) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                logits = model(data)
                
                loss = criterion(logits, target) # Correct as testing

                total_loss += loss.item()
                total_correct += (logits.argmax(dim=1) == target).sum().item()
                total_samples += data.size(0)
        # calculate average accuracy and loss
        total_loss /= (idx+1)
        total_acc = total_correct / total_samples

        if del_model:
            del model
        return total_loss, total_acc
    
    def test_with_task_id(self,task_id, model=None):
        task_id_device = copy.copy(torch.tensor(task_id)).to(self.device)
        del_model = False
        if model ==  None:
            model = copy.deepcopy(self.global_model)
            del_model = True
        model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        if "imagenet" in self.args.dataset:
            batch_size = 32
        else:
            batch_size = 128
        test_set = self.bm.test_stream[task_id].dataset
        test_loader = DataLoader(test_set, batch_size=batch_size)
        with torch.no_grad():
            for idx, (data, target, task_id_label) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                logits = model(data)
                loss = criterion(logits, target) 
    
                total_loss += loss.item()
                total_correct += (logits.argmax(dim=1) == target).sum().item()
                total_samples += data.size(0)

        total_loss /= (idx+1)
        total_acc = total_correct / total_samples

        if del_model:
            del model
        return total_loss, total_acc
    
    def train_metric_with_task_id(self, task_id, model=None):
        task_id_device = copy.copy(torch.tensor(task_id)).to(self.device)
        del_model = False
        if model ==  None:
            model = copy.deepcopy(self.global_model)
            del_model = True
        model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        # Fetch the train_loader based on task_id
        if "imagenet" in self.args.dataset:
            batch_size = 32
        else:
            batch_size = 128
        train_set = self.bm.train_stream[task_id].dataset
        train_loader = DataLoader(train_set, batch_size=batch_size)
        with torch.no_grad():
            for idx, (data, target, task_id_label) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                logits = model(data)
                loss = criterion(logits, target) # correct as testing
    
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
    print(f"Running script ***: {os.path.basename(__file__)}")
    parser = arg_parser()

    result_path = f'./saved_results_incremental/results_{parser.dataset}_{parser.model_name}/buffer_{parser.initial_buffer_size}/alpha_{parser.alpha}/lrs_{parser.lr}-{parser.lr_retrain}/momen_{parser.momentum}-{parser.momentum_retrain}/seed_{parser.seed}/'
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
                    class_ids_from_zero_from_first_exp=True,
                    train_transform=transform, eval_transform=transform
                    )
    elif parser.dataset == "PermutedMNIST":
        num_tasks = parser.num_tasks
        transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))  # Normalize inputs
                                        ])
        bm = PermutedMNIST(n_experiences=num_tasks,
                    return_task_id=True,
                    seed=parser.seed,
                    class_ids_from_zero_from_first_exp=True,
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
                    class_ids_from_zero_from_first_exp=True,
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
                    class_ids_from_zero_from_first_exp=True,
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
                    class_ids_from_zero_from_first_exp=True,
                    train_transform=transform_train, eval_transform=transform_test
                    )
    else:
        raise NotImplementedError
    print(f"Dataset: {parser.dataset} with splits: {num_tasks}")
    print(parser)
    algo = CFLAG(parser, bm, num_tasks,)
    algo.server()
    print(parser)
