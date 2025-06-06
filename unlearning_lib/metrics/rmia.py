import torch
import numpy as np
from scipy.stats import norm, trim_mean

def majority_voting_tensor(tensor, axis): # compute majority voting for a bool tensor along a certain axis 
    return torch.mode(torch.stack(tensor), axis).values * 1.0

def config_dataset(dataname='cifar10'):
    if dataname == 'cifar10' or dataname == 'cinic10':
        proptocut = float(0.2)
        offline_a = float(0.33)
        offline_b = float(0.66)
        extra = {}
        extra["taylor_m"] = float(0.6) # float
        extra["taylor_n"] = 4 # int
        temperature = float(2.0)
        gamma = float(2.0)

    elif dataname == 'cifar100':
        proptocut = float(0.2)
        offline_a = float(0.6)
        offline_b = float(0.4)
        extra = {}
        extra["taylor_m"] = float(0.6) # float
        extra["taylor_n"] = 4 # int
        temperature = float(1.0)
        gamma = float(2.0)

    elif dataname == 'location':
        proptocut = float(0.2)
        offline_a = float(0.2)
        offline_b = float(0.8)
        extra = {}
        extra["taylor_m"] = float(0.6) # float
        extra["taylor_n"] = 4 # int
        temperature = float(2.0)
        gamma = float(2.0)

    elif dataname == 'texas':
        proptocut = float(0.2)
        offline_a = float(0.2)
        offline_b = float(0.8)
        extra = {}
        extra["taylor_m"] = float(0.6) # float
        extra["taylor_n"] = 4 # int
        temperature = float(2.0)
        gamma = float(2.0)
        
    elif dataname == 'purchase':
        proptocut = float(0.2)
        offline_a = float(0.2)
        offline_b = float(0.8)
        extra = {}
        extra["taylor_m"] = float(0.6) # float
        extra["taylor_n"] = 4 # int
        temperature = float(2.0)
        gamma = float(2.0)

    else:
        raise ValueError(f"Unknown dataname: {dataname}")
    
    return proptocut, offline_a, offline_b, extra, temperature, gamma

def rmia(target_logit_target, ref_logit_target, target_label_target, 
         target_logit_population, ref_logit_population, target_label_population, 
         OFFLINE=True, ref_in_logit_target=None, ref_in_logit_pop=None, 
         model_numbers=16, model_list=None, metric = 'taylor-soft-margin', dataname=None, batch_size=256, num_workers=2, DEVICE='cpu'):
        
    proptocut, offline_a, offline_b, extra, temperature, gamma = config_dataset(dataname)
    ref_signal_targets, ref_signal_populations = [], []
    if model_list is None:
        model_list = list(range(model_numbers))
    for ref_idx in model_list:
        ref_signal_target = convert_signals(ref_logit_target[:,ref_idx,:].squeeze(), target_label_target, metric, temp=temperature, extra=extra)
        ref_signal_targets.append(ref_signal_target)
        ref_signal_population = convert_signals(ref_logit_population[:,ref_idx,:].squeeze(), target_label_population, metric, temp=temperature, extra=extra)
        ref_signal_populations.append(ref_signal_population)

    if OFFLINE:
        ref_signal_targets = torch.stack(ref_signal_targets)
        ref_signal_populations = torch.stack(ref_signal_populations)
        mean_x = trim_mean(ref_signal_targets, proportiontocut=proptocut, axis=0)
        mean_z = trim_mean(ref_signal_populations, proportiontocut=proptocut, axis=0)
    else:        
        ref_out_signal_targets = torch.stack(ref_signal_targets)
        ref_out_signal_populations = torch.stack(ref_signal_populations)
        # mean_x_out = ref_out_signal_targets.mean(dim=0)
        # mean_z_out = ref_out_signal_populations.mean(dim=0)

        ref_in_signal_target = convert_signals(ref_in_logit_target, target_label_target, metric, temp=temperature, extra=extra)
        ref_in_signal_population = convert_signals(ref_in_logit_pop, target_label_population, metric, temp=temperature, extra=extra)
        ref_in_signal_targets = ref_in_signal_target.unsqueeze(0).expand(ref_out_signal_targets.shape[0],-1)
        ref_in_signal_populations = ref_in_signal_population.unsqueeze(0).expand(ref_out_signal_populations.shape[0],-1)
        # mean_x_in = ref_in_signal_target
        # mean_z_in = ref_in_signal_population
        # ref_signal_targets = torch.stack([mean_x_out, mean_x_in])
        # ref_signal_populations = torch.stack([mean_z_out, mean_z_in])
        ref_signal_targets = torch.concat([ref_out_signal_targets, ref_in_signal_targets])
        ref_signal_populations = torch.concat([ref_out_signal_populations, ref_in_signal_populations])
        mean_x =  trim_mean(ref_signal_targets, proportiontocut=proptocut, axis=0)
        mean_z = trim_mean(ref_signal_populations, proportiontocut=proptocut, axis=0)

    target_signal_target = convert_signals(target_logit_target, target_label_target, metric, temp=temperature, extra=extra)
    target_signal_population = convert_signals(target_logit_population, target_label_population, metric, temp=temperature, extra=extra)



    if OFFLINE:
        prob_ratio_x = (target_signal_target.ravel() / ((1+offline_a)/2 * mean_x + (1-offline_a) /2))
        prob_ratio_z_rev = 1 / (target_signal_population.ravel() / ((1+offline_a)/2 * mean_z + (1-offline_a) /2)) # the inverse to compute quickly
    else:
        prob_ratio_x = (target_signal_target.ravel()  / (mean_x))
        prob_ratio_z_rev = 1 / (target_signal_population.ravel() / (mean_z)) # the inverse to compute quickly
    
    final_scores = torch.outer(prob_ratio_x, prob_ratio_z_rev)
    signal_gamma = ((final_scores > float(gamma) )*1.0).mean(1).reshape(len(mean_x))
    
    prediction = np.array(signal_gamma)
    return prediction


def factorial(n):
    fact = 1
    for i in range(2, n + 1):
        fact = fact * i
    return fact

def get_taylor(logit_signals, n):
    power = logit_signals
    taylor = power + 1.0
    for i in range(2, n):
        power = power * logit_signals
        taylor = taylor + (power / factorial(i))
    return taylor

def convert_signals(all_logits, all_true_labels, metric, temp, extra=None):
    if metric == 'softmax':
        logit_signals = torch.div(all_logits, temp)
        max_logit_signals, max_indices = torch.max(logit_signals, dim=1)
        logit_signals = torch.sub(logit_signals, max_logit_signals.reshape(-1, 1))
        exp_logit_signals = torch.exp(logit_signals)
        exp_logit_sum = exp_logit_signals.sum(axis=1).reshape(-1, 1)
        true_exp_logit = exp_logit_signals.gather(1, all_true_labels.reshape(-1, 1))
        output_signals = torch.div(true_exp_logit, exp_logit_sum)
    elif metric == 'taylor':
        n = extra["taylor_n"]
        taylor_signals = get_taylor(all_logits, n)
        taylor_logit_sum = taylor_signals.sum(axis=1).reshape(-1, 1)
        true_taylor_logit = taylor_signals.gather(1, all_true_labels.reshape(-1, 1))
        output_signals = torch.div(true_taylor_logit, taylor_logit_sum)
    elif metric == 'soft-margin':
        m = float(extra["taylor_m"])
        logit_signals = torch.div(all_logits, temp)
        exp_logit_signals = torch.exp(logit_signals)
        exp_logit_sum = exp_logit_signals.sum(axis=1).reshape(-1, 1)
        true_logits = logit_signals.gather(1, all_true_labels.reshape(-1, 1))
        exp_true_logit = exp_logit_signals.gather(1, all_true_labels.reshape(-1, 1))
        exp_logit_sum = exp_logit_sum - exp_true_logit
        soft_true_logit = torch.exp(true_logits - m)
        exp_logit_sum = exp_logit_sum + soft_true_logit
        output_signals = torch.div(soft_true_logit, exp_logit_sum)
    elif metric == 'taylor-soft-margin':
        m, n = float(extra["taylor_m"]), int(extra["taylor_n"])
        logit_signals = torch.div(all_logits, temp)
        taylor_logits = get_taylor(logit_signals, n)
        taylor_logit_sum = taylor_logits.sum(axis=1).reshape(-1, 1)
        true_logit = logit_signals.gather(1, all_true_labels.reshape(-1, 1))
        taylor_true_logit = taylor_logits.gather(1, all_true_labels.reshape(-1, 1))
        taylor_logit_sum = taylor_logit_sum - taylor_true_logit
        soft_taylor_true_logit = get_taylor(true_logit - m, n)
        taylor_logit_sum = taylor_logit_sum + soft_taylor_true_logit
        output_signals = torch.div(soft_taylor_true_logit, taylor_logit_sum)
    elif metric == 'logits':
        output_signals = all_logits
    elif metric == 'log-logit-scaling': 
        # Correct Logit signal used by LiRA from Membership Inference Attacks From First Principles https://arxiv.org/abs/2112.03570 
        # Taken and readapted from https://github.com/carlini/privacy/blob/better-mi/research/mi_lira_2021/score.py
        # Can be used to compute the loss as in https://github.com/yuan74/ml_privacy_meter/blob/2022_enhanced_mia/research/2022_enhanced_mia/plot_attack_via_reference_or_distill.py
        predictions = all_logits - torch.max(all_logits, dim=1, keepdim=True).values
        predictions = torch.exp(predictions)
        predictions = predictions/torch.sum(predictions,dim=1,keepdim=True)
        COUNT = predictions.shape[0]
        y_true = predictions[np.arange(COUNT),all_true_labels[:COUNT]]
        predictions[np.arange(COUNT),all_true_labels[:COUNT]] = 0
        y_wrong = torch.sum(predictions, dim=1)
        output_signals = (torch.log(y_true+1e-45) - torch.log(y_wrong+1e-45))  

    output_signals = torch.flatten(output_signals)
    return output_signals.cpu()