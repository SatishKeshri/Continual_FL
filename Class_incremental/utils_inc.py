from typing import Any, Dict, List
import argparse
import os
import copy
import torch
import numpy as np



def average_weights(weights: List[Dict[str, torch.Tensor]], state_dict=True) -> Dict[str, torch.Tensor]:
    if state_dict == False:
        weights = [model.state_dict() for model in weights]
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] += weights[i][key]
        if 'num_batches_tracked' in key:
            weights_avg[key] = weights_avg[key].true_divide(len(weights))
        else:
            weights_avg[key] = torch.div(weights_avg[key], len(weights)) 
    return weights_avg


def get_acc_forgetting(res_list:list, num_tasks=5):
    """"""
    max_acc = []
    # print("Length of res_list: ", len(res_list))
    if len(res_list) == num_tasks:
        print(f"Full training done for {num_tasks} tasks")
    else:
        print(f"Partial training done for {len(res_list)} tasks out of {num_tasks} tasks")
    
    for i, entry in enumerate(res_list):
        # print(res_list[i][i][1])
        max_acc.append(res_list[i][i][1])
    final_stats = res_list[-1]
    final_acc_list = [final_stats[i][1] for i in range(len(final_stats))]
    avg_acc = np.mean(final_acc_list)
    avg_forgetting = np.mean([max_acc[i] - final_acc_list[i] for i in range(len(max_acc) - 1)])
    print(f"Average accuracy: {avg_acc*100}, Average forgetting: {avg_forgetting*100}")

def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="../datasets/")
    parser.add_argument("--dataset", type=str, default="CIFAR10") #["MNIST", "PermutedMNIST", "CIFAR10", "CIFAR100", "tinyimagenet"]
    parser.add_argument("--model_name", type=str, default="resnet18") #["MLP", "resnet18", "resnet50"]
    
    
    parser.add_argument("--num_tasks", type=int, default=5) # 5,10,20 for 
    parser.add_argument("--seed", type=int, default=1234) #default=1,
    # parser.add_argument("--class_per_head", type=int, default=2)
    parser.add_argument("--num_classes_total", type=int, default=10)
    parser.add_argument("--L", type=float, default=5)#5

    parser.add_argument("--alpha", type=float, default=100000.0) #Non-iid: 0.1, IID: 100000.0
    
    parser.add_argument("--frac", type=float, default=1)
    
    parser.add_argument("--multi_head_flag", default=False, action='store_true') 
    parser.add_argument("--IAG_flag", default=False, action='store_true') 
    parser.add_argument("--IAG_batch_flag", default=False, action='store_true') 
    parser.add_argument("--delayed_grad_batch_flag", default=False, action='store_true')
    parser.add_argument("--grad_f_flag",default=False, action='store_true') 
    parser.add_argument("--adaptive_flag",default=False, action='store_true') 
    parser.add_argument("--adaptive_lr_batch_flag",default=False, action='store_true') 
    parser.add_argument("--adaptive_lr_step_flag",default=False, action='store_true') 
    parser.add_argument("--adaptive_lr_round_flag", default=False, action='store_true') 
    parser.add_argument("--kick_client", default=False, action='store_true') 
    parser.add_argument("--kick_memory", default=False, action='store_true') 
    parser.add_argument("--proxy_adaptive_lr",default=False, action='store_true') 
    parser.add_argument("--gamma_flag",default=False, action='store_true')
    parser.add_argument("--fedtrack_flag",default=False, action='store_true')
    parser.add_argument("--add_f_adapt",default=False, action='store_true')
    parser.add_argument("--add_f_adapt_inside",default=False, action='store_true')

    parser.add_argument("--n_rounds", type=int, default=1) 
    parser.add_argument("--n_client_epochs", type=int, default=2) 
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--initial_buffer_size", type=int, default=400) 
    parser.add_argument("--memory_sample_size", type=int, default=100) 
    
    parser.add_argument("--optim", type=str, default="adam")# adam", "sgd"
    parser.add_argument("--lr", type=float, default=0.0001) # 0.1, 0.0001
    parser.add_argument("--lr_retrain", type=float, default=0.00005) # 0.1, 0.0001
    parser.add_argument("--momentum", type=float, default=0.9)#0.5
    parser.add_argument("--momentum_retrain", type=float, default=0.7)#0.9
    parser.add_argument("--memory_scheme", type=str, default="class_balanced")
    
    parser.add_argument("--clip_grad_g_val", default=False, action='store_true')
    parser.add_argument("--clip_grad_f_val", default=False, action='store_true')
    parser.add_argument("--min_clip", type=float, default=-0.5)
    parser.add_argument("--max_clip", type=float, default=0.5)
    parser.add_argument("--clip_grad_norms", default=False, action='store_true') 
    parser.add_argument("--norm_threshold", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-8) 

    parser.add_argument("--device", type=int, default=0)

    return parser.parse_args()
