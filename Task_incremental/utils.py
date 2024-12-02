from typing import Any, Dict, List
import argparse
import os
import copy
import torch



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

def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="../datasets/")
    parser.add_argument("--dataset", type=str, default="MNIST") #["MNIST", "PermutedMNIST", "CIFAR100", "CIFAR10", "tinyimagenet"]
    parser.add_argument("--model_name", type=str, default="MLP") #["MLP", "resnet18", "resnet50"]
    
    parser.add_argument("--optim", type=str, default="adam")# adam", "sgd"
    parser.add_argument("--num_tasks", type=int, default=5) # 5,10,20 for 
    parser.add_argument("--seed", type=int, default=1234) #default=1,
    # parser.add_argument("--class_per_head", type=int, default=2) # ISSUE: change inside main function
    parser.add_argument("--num_classes_total", type=int, default=10)
    parser.add_argument("--L", type=float, default=5)

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

    parser.add_argument("--n_rounds", type=int, default=2) 
    parser.add_argument("--n_client_epochs", type=int, default=3) 
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--initial_buffer_size", type=int, default=400) 
    parser.add_argument("--memory_sample_size", type=int, default=150) 
    
    # parser.add_argument("--optim", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lr_retrain", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)#0.5
    parser.add_argument("--momentum_retrain", type=float, default=0.9)#0.9
    parser.add_argument("--memory_scheme", type=str, default="class_balanced")
    
    parser.add_argument("--clip_grad_g_val", default=False, action='store_true')
    parser.add_argument("--clip_grad_f_val", default=False, action='store_true')
    parser.add_argument("--min_clip", type=float, default=-0.5)
    parser.add_argument("--max_clip", type=float, default=0.5)
    parser.add_argument("--clip_grad_norms", default=False, action='store_true') 
    parser.add_argument("--norm_threshold", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-6)

    parser.add_argument("--device", type=int, default=0)

    return parser.parse_args()
