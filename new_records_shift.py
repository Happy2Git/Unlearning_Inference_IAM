import os
import torch
import argparse
import copy

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.stats import norm, laplace # Replaced invgauss with norm (normal distribution)

from shadow_training import shadow_training
from utils import get_unlearn_model
from data_loader import get_data_loaders

from unlearning_lib.models.resnet import ResNet18
from unlearning_lib.models.purchase_classifier import PurchaseClassifier
from unlearning_lib.metrics.feature_builder import carlini_logit, model_confidence, ce_loss
from unlearning_lib.metrics.auc_extended import auc_extended
from unlearning_lib.metrics.rmia import rmia, convert_signals, config_dataset
from unlearning_lib.metrics.enhanced_mia import enhanced_mia, enhanced_mia_p
from unlearning_lib.metrics.iam_score import interpolate_appro
from unlearning_lib.utils import load_pretrain_weights, accuracy, train_classifier, train_attack_model, construct_leak_feature

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", DEVICE.upper())

# if DEVICE != 'cuda':
#     raise RuntimeError('Make sure you have added an accelerator to your notebook; the submission will fail otherwise!')



def audit_nonmem_mono(args, DEVICE=torch.device("cpu"), SEED=42, batch_size=1024):
    dataname = args.dataname
    shadow_data = 'cifar10' if dataname == 'cinic10' else 'cinic10'    
    # Check if all logits are saved, if not, save them
    records_folder = args.records_folder + f'seed_{SEED}/'
    records_path = load_records(records_folder, args, batch_size=batch_size, DEVICE=DEVICE, SEED=SEED)
    # return

    # for metrics in args.metrics:
    #     if metrics not in ['p_Unleak', 'p_LiRA', 'p_update_LiRA', 'p_EMIA', 'p_RMIA', 'p_SimpleDiff', 'p_Dratio']:
    #         raise ValueError(f"Metrics {metrics} is not supported")

    audit_predicts = {}
    if 'p_Unleak' in args.metrics:
        # train attack models on nonmem set
        attack_model = train_attack_model(
            args, model_list=args.SHADOW_IDXs, shadow_path= args.shadow_folder + f'{shadow_data}', DEVICE=DEVICE)
        
        target_ori_probs = torch.nn.functional.softmax(torch.load(records_path['target_ori_logits']), dim=1)
        target_unl_probs = torch.nn.functional.softmax(torch.load(records_path['target_unl_logits']), dim=1)

        target_leak_feature = construct_leak_feature(
            target_ori_probs, target_unl_probs)
        p_Unleak = attack_model.predict_proba(
            target_leak_feature)[:, 1]
        audit_predicts['p_Unleak'] = p_Unleak
        print(f"p_Unleak done")

    if 'p_LiRA' in args.metrics or 'p_update_LiRA' in args.metrics:
        target_labels = torch.load(records_path['target_labels'])
        target_unl_logits = torch.load(records_path['target_unl_logits'])
        target_unl_carlogi = carlini_logit(target_unl_logits, target_labels)
        target_ori_logits = torch.load(records_path['target_ori_logits'])
        target_ori_carlogi = carlini_logit(target_ori_logits, target_labels)
        target_shadow_logits = torch.load(records_path['target_shadow_logits'])
        shadow_target_ori_carlogis = []
        for shadow_idx in args.SHADOW_IDXs:
            shadow_target_ori_carlogi = carlini_logit(target_shadow_logits[:, shadow_idx, :].squeeze() , target_labels)
            shadow_target_ori_carlogis.append(shadow_target_ori_carlogi)

        shadow_target_ori_carlogis = torch.stack(shadow_target_ori_carlogis, dim=1).numpy()
        mean_shadow_target_out_carlogis = shadow_target_ori_carlogis.mean(axis=1)
        std_shadow_target_out_carlogis = shadow_target_ori_carlogis.std(axis=1)
        fix_variance = False
        if fix_variance:
            std_shadow_fix = shadow_target_ori_carlogis.std()
            std_shadow_target_out_carlogis = np.full_like(std_shadow_target_out_carlogis, std_shadow_fix)
    
        if 'p_LiRA' in args.metrics:
            if args.unlearn_type == 'set_random':
                p_LiRA = -norm.logpdf(target_unl_carlogi, loc=mean_shadow_target_out_carlogis,
                                        scale=std_shadow_target_out_carlogis+1e-30)
            else:
                p_LiRA = norm.cdf(target_unl_carlogi, loc=mean_shadow_target_out_carlogis,
                        scale=std_shadow_target_out_carlogis+1e-30)
            audit_predicts['p_LiRA'] = -p_LiRA

            mean_shadow_target_in_carlogis = target_ori_carlogi.numpy()
            std_shadow_target_in_carlogis = torch.zeros_like(target_ori_carlogi).numpy()
            if fix_variance:
                std_shadow_fix = target_ori_carlogi.numpy().std()
                std_shadow_target_in_carlogis = np.full_like(std_shadow_target_in_carlogis, std_shadow_fix)

            p_in = -norm.logpdf(target_unl_carlogi, loc=mean_shadow_target_in_carlogis,
                                scale=std_shadow_target_in_carlogis+1e-30)
            p_out = -norm.logpdf(target_unl_carlogi, loc=mean_shadow_target_out_carlogis,
                                    scale=std_shadow_target_out_carlogis+1e-30)
            p_LiRA_Online = (p_in - p_out)

            audit_predicts['p_LiRA_Online'] = p_LiRA_Online
            print(f"p_LiRA done")

        if 'p_update_LiRA' in args.metrics:
            target_ori_carlogi = carlini_logit(torch.load(records_path['target_ori_logits']), target_labels)
            p_out_LiRA = 1-norm.cdf(target_unl_carlogi, loc=mean_shadow_target_out_carlogis,
                                    scale=std_shadow_target_out_carlogis+1e-30)
            p_in_LiRA = 1- norm.cdf(target_ori_carlogi, loc=mean_shadow_target_out_carlogis,
                                scale=std_shadow_target_out_carlogis+1e-30)
            p_update_LiRA = p_out_LiRA - p_in_LiRA
            audit_predicts['p_update_LiRA'] = p_update_LiRA
            print(f"p_update_LiRA done")

        # shift_score = target_ori_carlogi - shadow_target_ori_carlogis.T
        # observe = target_unl_carlogi - shadow_target_ori_carlogis.mean(dim=1)
        # means = shift_score.mean(dim=0)
        # stds = shift_score.std(dim=0)
        # p_non_LiRA = 1-norm.cdf(observe, loc=means, scale=stds+1e-30)
        # audit_predicts['p_non_LiRA_new'] = p_non_LiRA
        # print(f"p_non_LiRA_new done")

    if 'p_EMIA' in args.metrics:
        target_labels = torch.load(records_path['target_labels'])
        target_unl_logits = torch.load(records_path['target_unl_logits'])
        target_unl_logity = target_unl_logits[range(len(target_labels)), target_labels]
        target_shadow_logits = torch.load(records_path['target_shadow_logits'])
        target_shadow_logitsy = target_shadow_logits[range(len(target_labels)), :, target_labels].squeeze().T
        if target_shadow_logits.shape[1]==1:
            target_shadow_logitsy = target_shadow_logitsy.unsqueeze(0)

        p_EMIA = enhanced_mia(target_unl_logity, target_shadow_logitsy[args.SHADOW_IDXs,:])
        audit_predicts['p_EMIA'] = p_EMIA
        print(f"p_EMIA done")

    if 'p_EMIA_p' in args.metrics:
        target_labels = torch.load(records_path['target_labels'])
        target_unl_logits = torch.load(records_path['target_unl_logits'])
        target_unl_logity = target_unl_logits[range(len(target_labels)), target_labels]
        ref_unl_logits = torch.load(records_path['ref_unl_logits'])
        ref_labels = torch.load(records_path['ref_labels'])
        ref_unl_logitsy = ref_unl_logits[range(len(ref_labels)), ref_labels].squeeze().T
        p_EMIA = enhanced_mia_p(target_unl_logity, ref_unl_logitsy)
        audit_predicts['p_EMIA_p'] = p_EMIA
        print(f"p_EMIA_p done")

    if 'p_random' in args.metrics:
        target_labels = torch.load(records_path['target_labels'])
        p_random = torch.rand_like(target_labels.float()).cpu().numpy()
        audit_predicts['p_random'] = p_random
        print(f"p_random done")

    if 'p_RMIA' in args.metrics:
        target_unl_logits = torch.load(records_path['target_unl_logits'])
        target_ori_logits = torch.load(records_path['target_ori_logits'])
        target_labels = torch.load(records_path['target_labels'])
        ref_unl_logits = torch.load(records_path['ref_unl_logits'])
        ref_labels = torch.load(records_path['ref_labels'])
        target_shadow_logits = torch.load(records_path['target_shadow_logits'])
        ref_shadow_logits = torch.load(records_path['ref_shadow_logits'])
        # print(f'ref_shadow_logits shape: {ref_shadow_logits.shape}')
        # print(f'path{records_path}')

        p_RMIA = 1-rmia(target_unl_logits, target_shadow_logits, target_labels, 
                ref_unl_logits, ref_shadow_logits, ref_labels, 
                model_list=args.SHADOW_IDXs, metric='taylor-soft-margin', dataname= args.dataname, DEVICE=DEVICE)

        audit_predicts['p_RMIA'] = p_RMIA

        ref_in_logit_pop = torch.load(records_path['ref_popin_logits'])

        p_RMIA = 1-rmia(target_unl_logits, target_shadow_logits, target_labels, 
                ref_unl_logits, ref_shadow_logits, ref_labels, 
                OFFLINE=False, ref_in_logit_target=target_ori_logits, ref_in_logit_pop=ref_in_logit_pop,
                model_list=args.SHADOW_IDXs, metric='taylor-soft-margin', dataname= args.dataname, DEVICE=DEVICE)

        audit_predicts['p_RMIA_online'] = p_RMIA

        print(f"p_RMIA done")

    if 'p_SimpleDiff' in args.metrics:
        target_ori_logits = torch.load(records_path['target_ori_logits'])
        target_unl_logits = torch.load(records_path['target_unl_logits'])
        target_labels = torch.load(records_path['target_labels'])

        target_unl_proby = model_confidence(target_unl_logits, target_labels)
        target_ori_proby = model_confidence(target_ori_logits, target_labels)
        p_SimpleDiff_conf = target_ori_proby - target_unl_proby
        p_SimpleDiff_conf = target_ori_proby - target_unl_proby
        audit_predicts['p_SimpleDiff_conf'] = p_SimpleDiff_conf.numpy()
        print(f"p_SimpleDiff_conf done")

        audit_predicts['p_Simple_conf'] = -target_unl_proby.numpy()
        print(f"p_Simple_conf done")

        target_unl_loss = ce_loss(target_unl_logits, target_labels)
        target_ori_loss = ce_loss(target_ori_logits, target_labels)
        p_SimpleLoss = target_unl_loss - target_ori_loss
        audit_predicts['p_SimpleLoss'] = p_SimpleLoss.numpy()
        print(f"p_SimpleLoss done")

        target_unl_carlogi = carlini_logit(target_unl_logits, target_labels)
        target_ori_carlogi = carlini_logit(target_ori_logits, target_labels)
        p_SimpleCarlogi = target_ori_carlogi - target_unl_carlogi
        audit_predicts['p_SimpleCarlogi'] = p_SimpleCarlogi.numpy()
        print(f"p_SimpleCarlogi done")


        '''
        becaus ethe celoss has been on a logaritmic scale, so we use logit = exp(-exp(score)), 
        and we use the log of the logloss to get a distribution can be fitted by gumbel_r distribution
        '''
        eps1 = 1e-2
        eps2 = 1e-5
        max_value =  -np.log(eps1-np.log(eps2+1))
        min_value = -np.log(eps1-np.log(eps2+0))
        target_unl_logloss = (-np.log(eps1-np.log(eps2+target_unl_proby))-min_value)/(max_value-min_value)
        target_ori_logloss = (-np.log(eps1-np.log(eps2+target_ori_proby))-min_value)/(max_value-min_value)
        p_SimpleLogloss = target_ori_logloss - target_unl_logloss
        audit_predicts['p_SimpleLogloss'] = p_SimpleLogloss
        print(f"p_SimpleLogloss done")


    if 'p_LDiff' in args.metrics:
        # load logits and get logloss
        target_ori_logits = torch.load(records_path['target_ori_logits'])
        target_unl_logits = torch.load(records_path['target_unl_logits'])
        target_labels = torch.load(records_path['target_labels'])

        eps1= 1e-2 if args.dataname != 'texas' else 1e-1
        eps2 = 1e-5
        max_value =  -np.log(eps1-np.log(eps2+1))
        min_value = -np.log(eps1-np.log(eps2+0))
        target_unl_logloss = (-np.log(eps1-np.log(eps2+target_unl_proby))-min_value)/(max_value-min_value)
        target_ori_logloss = (-np.log(eps1-np.log(eps2+target_ori_proby))-min_value)/(max_value-min_value)

        ref_ori_logits = torch.load(records_path['ref_ori_logits'])
        ref_unl_logits = torch.load(records_path['ref_unl_logits'])
        ref_labels = torch.load(records_path['ref_labels'])

        ref_ori_proby = model_confidence(ref_ori_logits, ref_labels)
        ref_unl_proby = model_confidence(ref_unl_logits, ref_labels)

        ref_ori_logloss = (-np.log(eps1-np.log(eps2+ref_ori_proby))-min_value)/(max_value-min_value)
        ref_unl_logloss =  (-np.log(eps1-np.log(eps2+ref_unl_proby))-min_value)/(max_value-min_value)

        target_ref_logits = torch.load(records_path['target_shadow_logits'])
        shadow_ref_logits = torch.load(records_path['ref_shadow_logits'])

        target_ref_logits = torch.load(records_path['target_shadow_logits'])

        ref_ori_carlogi = carlini_logit(ref_ori_logits, ref_labels)
        ref_unl_carlogi = carlini_logit(ref_unl_logits, ref_labels)
        mean_old, std_old = torch.mean(ref_ori_carlogi), torch.std(ref_ori_carlogi)
        mean_new, std_new = torch.mean(ref_unl_carlogi), torch.std(ref_unl_carlogi)
        target_unl_carlogi = carlini_logit(target_unl_logits, target_labels)
        target_ori_carlogi = carlini_logit(target_ori_logits, target_labels)
        lira_score_unl = 1-norm.cdf(
            target_unl_carlogi, loc=mean_new, scale=std_new+1e-30)
        lira_score_ori = 1-norm.cdf(
            target_ori_carlogi, loc=mean_new, scale=std_new+1e-30)
        lira_diff = lira_score_unl - lira_score_ori
        lira_diff = (lira_diff + 1)/2
        ref_diff_carlogi = ref_unl_carlogi - ref_ori_carlogi
        target_diff_carlogi = target_unl_carlogi - target_ori_carlogi
        para_laplace= laplace.fit(ref_diff_carlogi)
        diff_lira = 1-laplace.cdf(target_diff_carlogi, *para_laplace)
        p_ave_LnDl = (lira_diff+diff_lira)/2
        audit_predicts['p_old_LnDl'] = p_ave_LnDl

        # shadow_target_logloss = []
        # for ref_idx in args.SHADOW_IDXs:
        #     shadow_signal_target = model_confidence(target_ref_logits[:,ref_idx,:].squeeze(), target_labels)
        #     shadow_target_logloss.append((-np.log(eps1-np.log(eps2+shadow_signal_target))-min_value)/(max_value-min_value))
        # shadow_target_logloss = torch.stack(shadow_target_logloss)

        # # start to calculate the p_LDiff
        # const = 0.001
        # diff_logloss = target_ori_logloss-target_unl_logloss
        # over_logloss = target_ori_logloss - shadow_target_logloss.mean(dim=0)
        # diff_logloss = torch.where(diff_logloss<0, torch.full_like(diff_logloss, 1e-10), diff_logloss)
        # over_logloss = torch.where(over_logloss<0, torch.full_like(over_logloss, 1e-10), over_logloss)
        # diff_ratio =  (const+diff_logloss)/(const+over_logloss)
        # diff_ratio = torch.where(diff_ratio>1, torch.full_like(diff_ratio, 1), diff_ratio)
        # diff_ratio = torch.where(diff_ratio<0, torch.full_like(diff_ratio, 0), diff_ratio)
        # key = 'logloss_diff_ratio'
        # audit_predicts[key] = diff_ratio
        # print(f"{key} done")

        # diff_logloss = target_unl_logloss - shadow_target_logloss
        # over_logloss = target_ori_logloss - shadow_target_logloss
        # diff_logloss = torch.where(diff_logloss<0, torch.full_like(diff_logloss, 1e-10), diff_logloss)
        # over_logloss = torch.where(over_logloss<0, torch.full_like(over_logloss, 1e-10), over_logloss)
        # diff_ratio = 1-((const+diff_logloss)/(const+over_logloss)).mean(dim=0)
        # diff_ratio = torch.where(diff_ratio>1, torch.full_like(diff_ratio, 1), diff_ratio)
        # diff_ratio = torch.where(diff_ratio<0, torch.full_like(diff_ratio, 0), diff_ratio)

        # key = 'logloss_diff_ratio_minus'
        # audit_predicts[key] = diff_ratio
        # print(f"{key} done")

        # discrete_interp = np.linspace(target_ori_logloss, shadow_target_logloss, 100)
        # vote_interp = (discrete_interp>target_unl_logloss)*1.0
        # vote_score = vote_interp.view(-1, vote_interp.shape[-1]).mean(dim=0)
        # key = 'diff_vote'
        # audit_predicts[key] = vote_score
        # print(f"{key} done")

        # # gumbel_nulllike
        # mean_data = (shadow_target_logloss).mean(dim=0)
        # std_data =(shadow_target_logloss).std(dim=0)
        # # Estimate parameters using method of moments
        # gamma = 0.5772  # Euler-Mascheroni constant
        # beta_mom = np.sqrt(6 * std_data) / np.pi
        # mu_mom = mean_data - beta_mom * gamma
        # score_gumbel_ori = gumbel_r.sf(target_ori_logloss, mu_mom, beta_mom)
        # score_gumbel_unl = gumbel_r.sf(target_unl_logloss, mu_mom, beta_mom)
        # gumbel_nulllike_diff = score_gumbel_unl - score_gumbel_ori
        # key = 'gumbel_nulllike_diff'
        # audit_predicts[key] = gumbel_nulllike_diff
        # print(f"{key} done")


        # # shift gumbel_nulllike by shadow_ref, ref_ori, ref_unl
        # shadow_ref_logloss = []
        # eps = 0.01
        # for ref_idx in args.SHADOW_IDXs:
        #     shadow_signal_ref = model_confidence(shadow_ref_logits[:,ref_idx,:].squeeze(), ref_labels)
        #     shadow_ref_logloss.append((-np.log(eps1-np.log(eps2+shadow_signal_ref))-min_value)/(max_value-min_value))
        # shadow_ref_logloss = torch.stack(shadow_ref_logloss)

        # model_level_gen_gen_gap = (ref_ori_logloss- shadow_ref_logloss).mean(dim=1)
        # mean_data = (shadow_target_logloss+model_level_gen_gen_gap.unsqueeze(1)).mean(dim=0)        
        # std_data =(shadow_target_logloss).std(dim=0)
        # # Estimate parameters using method of moments
        # gamma = 0.5772  # Euler-Mascheroni constant
        # beta_mom = np.sqrt(6 * std_data) / np.pi
        # mu_mom = mean_data - beta_mom * gamma
        # score_gumbel_ori_shift2target = gumbel_r.sf(target_ori_logloss, mu_mom, beta_mom)
        # score_gumbel_unl_shift2target = gumbel_r.sf(target_unl_logloss, mu_mom, beta_mom)
        # gumbel_nulllike_diff_shift = score_gumbel_unl_shift2target - score_gumbel_ori_shift2target
        # key = 'gumbel_nulllike_diff_shift'
        # audit_predicts[key] = gumbel_nulllike_diff_shift
        # print(f"{key} done")


    if 'p_interapprox' in args.metrics:
        INTER_APPRO = args.INTERAPPROX
        target_ori_logits = torch.load(records_path['target_ori_logits'])
        target_unl_logits = torch.load(records_path['target_unl_logits'])
        target_labels = torch.load(records_path['target_labels'])

        target_unl_proby = model_confidence(target_unl_logits, target_labels)
        target_ori_proby = model_confidence(target_ori_logits, target_labels)
        target_ref_logits = torch.load(records_path['target_shadow_logits'])
        shadow_target_proby = []
        for ref_idx in args.SHADOW_IDXs:
            shadow_signal_target = model_confidence(target_ref_logits[:,ref_idx,:].squeeze(), target_labels)
            shadow_target_proby.append((shadow_signal_target))
        shadow_target_proby = torch.stack(shadow_target_proby)
        score_ori_weighted, score_unl_weighted = interpolate_appro(target_ori_proby, target_unl_proby, shadow_target_proby, INTER_APPRO, type = 'norm')
        key = 'p_IPapprox_proby'
        audit_predicts[key] = score_unl_weighted - score_ori_weighted
        key = 'p_IPapprox_proby_simp'
        audit_predicts[key] = score_unl_weighted
        print(f"{key} done")

        _, _, _, extra, temperature, gamma = config_dataset(args.dataname)
        target_unl_tsm = convert_signals(target_unl_logits, target_labels, 'taylor-soft-margin', temp=temperature, extra=extra)
        target_ori_tsm = convert_signals(target_ori_logits, target_labels, 'taylor-soft-margin', temp=temperature, extra=extra)
        target_ref_logits = torch.load(records_path['target_shadow_logits'])
        shadow_target_tsm = []
        for ref_idx in args.SHADOW_IDXs:
            shadow_signal_target = convert_signals(target_ref_logits[:,ref_idx,:].squeeze(), target_labels, 'taylor-soft-margin', temp=temperature, extra=extra)
            shadow_target_tsm.append((shadow_signal_target))
        shadow_target_tsm = torch.stack(shadow_target_tsm)
        score_ori_weighted, score_unl_weighted = interpolate_appro(target_ori_tsm, target_unl_tsm, shadow_target_tsm, INTER_APPRO, type = 'norm')
        key = 'p_IPapprox_tsm'
        audit_predicts[key] = score_unl_weighted - score_ori_weighted
        key = 'p_IPapprox_tsm_simp'
        audit_predicts[key] = score_unl_weighted
        print(f"{key} done")

        target_unl_carlogi = carlini_logit(target_unl_logits, target_labels)
        target_ori_carlogi = carlini_logit(target_ori_logits, target_labels)
        target_ref_logits = torch.load(records_path['target_shadow_logits'])
        shadow_target_carlogi = []
        for ref_idx in args.SHADOW_IDXs:
            shadow_signal_target = carlini_logit(target_ref_logits[:,ref_idx,:].squeeze(), target_labels)
            shadow_target_carlogi.append((shadow_signal_target))
        shadow_target_carlogi = torch.stack(shadow_target_carlogi)
        score_ori_weighted, score_unl_weighted = interpolate_appro(target_ori_carlogi, target_unl_carlogi, shadow_target_carlogi, INTER_APPRO, type = 'norm')
        key = 'p_IPapprox_carlogi'
        audit_predicts[key] = score_unl_weighted - score_ori_weighted
        key = 'p_IPapprox_carlogi_simp'
        audit_predicts[key] = score_unl_weighted
        print(f"{key} done")

        target_unl_loss = ce_loss(target_unl_logits, target_labels)
        target_ori_loss = ce_loss(target_ori_logits, target_labels)
        shadow_target_loss = []
        for ref_idx in args.SHADOW_IDXs:
            shadow_signal_target = ce_loss(target_ref_logits[:,ref_idx,:].squeeze(), target_labels)
            shadow_target_loss.append((shadow_signal_target))
        shadow_target_loss = torch.stack(shadow_target_loss)
        score_ori_weighted, score_unl_weighted = interpolate_appro(-target_ori_loss, -target_unl_loss, -shadow_target_loss, INTER_APPRO, type = 'norm')
        key = 'p_IPapprox_losy'
        audit_predicts[key] = score_unl_weighted - score_ori_weighted
        key = 'p_IPapprox_losy_simp'
        audit_predicts[key] = score_unl_weighted
        print(f"{key} done")

        eps1 = 1e-2
        eps2 = 1e-5
        max_value =  -np.log(eps1-np.log(eps2+1))
        min_value = -np.log(eps1-np.log(eps2+0))
        target_unl_logloss = (-np.log(eps1-np.log(eps2+target_unl_proby))-min_value)/(max_value-min_value)
        target_ori_logloss = (-np.log(eps1-np.log(eps2+target_ori_proby))-min_value)/(max_value-min_value)
        shadow_target_logloss = (-np.log(eps1-np.log(eps2+shadow_target_proby))-min_value)/(max_value-min_value)
        _, score_unl_weighted = interpolate_appro(target_ori_logloss, target_unl_logloss, shadow_target_logloss, INTER_APPRO)
        key = 'p_IPapprox_simp_online'
        audit_predicts[key] = score_unl_weighted
        print(f"{key} done")

        target_ori_signal_off = torch.load(records_path['target_ori_signal_off'])
        _, score_unl_weighted = interpolate_appro(target_ori_signal_off, target_unl_logloss, shadow_target_logloss, INTER_APPRO)
        key = 'p_IPapprox_simp_offline'
        audit_predicts[key] = score_unl_weighted
        print(f"{key} done")

        target_unl_logloss = (-np.log(eps1-np.log(eps2+target_unl_proby)))
        target_ori_logloss = (-np.log(eps1-np.log(eps2+target_ori_proby)))
        shadow_target_logloss = (-np.log(eps1-np.log(eps2+shadow_target_proby)))
        _, score_unl_weighted = interpolate_appro(target_ori_logloss, target_unl_logloss, shadow_target_logloss, INTER_APPRO)
        key = 'p_IPapprox_simp_online_wo_norm'
        audit_predicts[key] = score_unl_weighted
        print(f"{key} done")
    target_unlearn_flags = torch.load(records_path['target_unlearn_flags'])
    return audit_predicts, target_unlearn_flags, target_unl_proby

# Sigmoid-like transformation for both A and B
def sigmoid_like_transform(x, p=4):
    return (x**p) / (x**p + (1 - x)**p)

def load_ref_in_model(model_path, nonmem_set, test_set, batch_size=256, DEVICE='cpu'):
    dataname = model_path.split('/')[-2]
    if dataname == 'cifar10' or dataname == 'cinic10':
        num_classes = 10
    elif dataname == 'location':
        num_classes = 30
    elif dataname == 'cifar100' or dataname == 'texas' or dataname == 'purchase':
        num_classes = 100

    if dataname == 'cifar10' or dataname == 'cifar100' or dataname == 'cinic10':
        ref_in_model = ResNet18(num_classes=num_classes)
        lr = 0.1
        epochs = 200
    elif dataname == 'location':
        ref_in_model = LocationClassifier()
        lr = 0.01
        epochs = 100
    elif dataname == 'texas':
        ref_in_model = TexasClassifier()
        lr = 0.01
        epochs = 100
    elif dataname == 'purchase':
        ref_in_model = PurchaseClassifier()
        lr = 0.01
        epochs = 100
        
    ref_in_model_name = model_path + f'{dataname}_ref_in_model.pth'
    print(ref_in_model_name)
    nonmem_loader_shuffle = DataLoader(nonmem_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader_shuffle = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)
    if os.path.exists(ref_in_model_name):
        ref_in_model.load_state_dict(torch.load(ref_in_model_name))
        ref_in_model.to(DEVICE)
        ref_in_model.eval()
        test_acc = accuracy(ref_in_model, test_loader_shuffle, DEVICE)
        train_acc = accuracy(ref_in_model, nonmem_loader_shuffle, DEVICE)
        print(f"Test acc of ref_in_model: {test_acc}, Train acc of ref_in_model: {train_acc}")
    else:
        ref_in_model.to(DEVICE)
        ref_in_model.train()
        # define optimizer
        optimizer = torch.optim.SGD(ref_in_model.parameters(), lr=lr,
                            momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        criterion = torch.nn.CrossEntropyLoss()

        best_acc = 0
        for epoch in range(epochs):
            ref_in_model.train()
            for inputs, targets in nonmem_loader_shuffle:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = ref_in_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                # print("loss: ", loss)
                optimizer.step()
            scheduler.step()
            ref_in_model.eval()
            acc = accuracy(ref_in_model, test_loader_shuffle, DEVICE)
            print(f"Epoch: {epoch}, Test acc of ref_in_model: {acc}")
            if acc > best_acc:
                state = {
                    'net': ref_in_model.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                best_acc = acc

        torch.save(state['net'], ref_in_model_name)
        ref_in_model.load_state_dict(state['net'])
        ref_in_model.eval()
        test_acc = accuracy(ref_in_model, test_loader_shuffle, DEVICE)
        train_acc = accuracy(ref_in_model, nonmem_loader_shuffle, DEVICE)
        print(f"Test acc of ref_in_model: {test_acc}, Train acc of ref_in_model: {train_acc}")
    return ref_in_model

def load_records(records_folder, args, batch_size=1024, DEVICE='cpu', SEED='42'):

    if not os.path.exists(records_folder):
        os.makedirs(records_folder)
    if not os.path.exists(args.records_folder + 'share_ori/'):
        os.makedirs(args.records_folder + 'share_ori/')
    shadow_numbs = 32

    dataname = args.dataname
    shadow_data = 'cifar10' if dataname == 'cinic10' else 'cinic10'
    unlearn_method = args.unlearn_method
    unlearn_type = args.unlearn_type

    target_ori_signal_off_suffix = f'{dataname}_off_fitting_shift_{len(args.SHADOW_IDXs)}_'+'_'.join(map(str, args.SHADOW_IDXs))+'.pth'
    if len(target_ori_signal_off_suffix) > 200:
        target_ori_signal_off_suffix = target_ori_signal_off_suffix.split('.pkl')[0][:200]+'.pth'

    SUFFIX = args.SUFFIX
    print(SUFFIX)
    print(DEVICE)
    target_ori_logits_path = args.records_folder + f'share_ori/{dataname}_target_ori_logits.pth' # #number x #class
    target_ori_signal_off_path = args.records_folder + f'share_ori/' + target_ori_signal_off_suffix # #number x #class
    target_unl_logits_path = records_folder + f'{SUFFIX}_target_unl_logits.pth' # #number x #class
    target_labels_path = args.records_folder + f'share_ori/{dataname}_target_labels.pth'  # #number
    target_unlearn_flags_path = records_folder + f'{SUFFIX}_target_unlearn_flags.pth'  # #number
    target_shadow_logits_path = args.records_folder + f'share_ori/{dataname}_shift_{shadow_data}_target_shadows_logits.pth' # #number x #shadow x #class
    ref_ori_logits_path = records_folder + f'{dataname}_ref_ori_logits.pth' # #number x #class
    ref_unl_logits_path = records_folder + f'{SUFFIX}_ref_unl_logits.pth' # #number x #class
    ref_labels_path = records_folder + f'{dataname}_ref_labels.pth' # #number x #class
    ref_shadow_logits_path = records_folder + f'{dataname}_shift_{shadow_data}_ref_shadow_logits.pth' # #number x #ref x #class
    accuracy_path = records_folder + f'{SUFFIX}_accuracy.pth' # #number x #ref x #class
    ref_popin_logits_path = records_folder + f'{dataname}_ref_popin_logits.pth' # #number x #class
    if not os.path.exists(args.records_folder + 'Appro/'):
        os.makedirs(args.records_folder + 'Appro/')
    INTER_APPRO = 10
    SHADOW_APPRO = 16
    # target_appro_scales_path = args.records_folder + f'Appro/{dataname}_target_appro_{INTER_APPRO}_{SHADOW_APPRO}_scales.pth' # #number x #class
    # target_appro_logits_path = args.records_folder  + f'Appro/{dataname}_target_appro_{INTER_APPRO}_{SHADOW_APPRO}_logits.pth' # #number x #class

    records_path = {
        'target_ori_logits': target_ori_logits_path,
        'target_ori_signal_off': target_ori_signal_off_path,
        'target_unl_logits': target_unl_logits_path,
        'target_labels': target_labels_path,
        'target_unlearn_flags': target_unlearn_flags_path,
        'ref_ori_logits': ref_ori_logits_path,
        'ref_unl_logits': ref_unl_logits_path,
        'ref_labels': ref_labels_path,
        'target_shadow_logits': target_shadow_logits_path,
        'ref_shadow_logits': ref_shadow_logits_path,
        'accuracy_path': accuracy_path,
        # 'target_appro_scales': target_appro_scales_path,
        # 'target_appro_logits': target_appro_logits_path,
        'ref_popin_logits': ref_popin_logits_path
    }
    # check if all records are saved
    has_saved = True
    for key in records_path.keys():
        if not os.path.exists(records_path[key]):
            has_saved = False
            break

    if has_saved:
        if os.path.exists(records_path['ref_shadow_logits']):
            ref_shadow_logits = torch.load(records_path['ref_shadow_logits'])
            if ref_shadow_logits.shape[1] == shadow_numbs:
                return records_path
    
    print(f"Start to check and generate records from {records_path}")
    
    ori_model, unl_model, target_set, unlearn_flags, ref_set, test_loader, shadow_path, shadow_set, SUFFIX, accuracy = fetch_data_model(args, verbose=True, SEED=SEED, DEVICE=DEVICE)

    if not os.path.exists(accuracy_path):
        torch.save(accuracy, accuracy_path)
        print(f"Save accuracy to {accuracy_path}")

    if not os.path.exists(target_ori_logits_path) or not os.path.exists(target_unl_logits_path) or not os.path.exists(target_labels_path) or not os.path.exists(target_unlearn_flags_path):
        target_loader = DataLoader(target_set, batch_size=batch_size, shuffle=False, num_workers=4)
        target_ori_logits, target_unl_logits, target_labels = [], [], []
        ori_model.to(DEVICE)
        unl_model.to(DEVICE)
        ori_model.eval()
        unl_model.eval()
        with torch.no_grad():
            for x, y in target_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                target_ori_logit = ori_model(x)
                target_unl_logit = unl_model(x)
                target_ori_logits.append(target_ori_logit.cpu())
                target_unl_logits.append(target_unl_logit.cpu())
                target_labels.append(y.cpu())

        target_ori_logits = torch.cat(target_ori_logits)
        target_unl_logits = torch.cat(target_unl_logits)
        target_labels = torch.cat(target_labels)
        target_unlearn_flags = unlearn_flags

        torch.save(target_ori_logits, target_ori_logits_path)
        torch.save(target_unl_logits, target_unl_logits_path)
        torch.save(target_labels, target_labels_path)
        torch.save(target_unlearn_flags, target_unlearn_flags_path)
        print(f"Save target_ori/unl_logits/label/flag_path to {records_folder}")

    if not os.path.exists(target_ori_signal_off_path):
        eps1 = 1e-2
        eps2 = 1e-5
        max_value =  -np.log(eps1-np.log(eps2+1))
        min_value = -np.log(eps1-np.log(eps2+0))

        shadow_fit = 0
        shadow_path= args.shadow_folder + f'{shadow_data}'
        for shadow_idx in args.SHADOW_IDXs:
            load_path_logit_train = shadow_path + f'/shadow_model_{shadow_idx}_logit_train.pth'
            load_path_label_train = shadow_path + f'/shadow_model_{shadow_idx}_label_train.pth'
            shadow_local_logit_train = torch.load(load_path_logit_train, map_location=torch.device('cpu'))
            shadow_local_label_train = torch.load(load_path_label_train, map_location=torch.device('cpu'))
            shadow_fit_probs = model_confidence(shadow_local_logit_train, shadow_local_label_train)
            shadow_fit_logloss = (-np.log(eps1-np.log(eps2+shadow_fit_probs))-min_value)/(max_value-min_value)
            shadow_fit += shadow_fit_logloss.mean()

        shadow_fit /= len(args.SHADOW_IDXs)
        target_labels = torch.load(target_labels_path)
        target_ori_signal_off = shadow_fit*torch.ones_like(target_labels)
        torch.save(target_ori_signal_off, target_ori_signal_off_path)
        print(f'save IAM offline fitting signal to {target_ori_signal_off_path}')

    if not os.path.exists(ref_ori_logits_path) or not os.path.exists(ref_unl_logits_path) or not os.path.exists(ref_labels_path):
        ref_loader = DataLoader(ref_set, batch_size=batch_size, shuffle=False, num_workers=4)
        ref_ori_logits, ref_unl_logits, ref_labels = [], [], []
        ori_model.to(DEVICE)
        unl_model.to(DEVICE)
        ori_model.eval()
        unl_model.eval()
        with torch.no_grad():
            for x, y in ref_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                ref_ori_logit = ori_model(x)
                ref_unl_logit = unl_model(x)
                ref_ori_logits.append(ref_ori_logit.cpu())
                ref_unl_logits.append(ref_unl_logit.cpu())
                ref_labels.append(y.cpu())

        ref_ori_logits = torch.cat(ref_ori_logits)
        ref_unl_logits = torch.cat(ref_unl_logits)
        ref_labels = torch.cat(ref_labels)
        torch.save(ref_ori_logits, ref_ori_logits_path)
        torch.save(ref_unl_logits, ref_unl_logits_path)
        torch.save(ref_labels, ref_labels_path)
        print(f"Save ref_ori/unl_logits/label_path to {records_folder}")
    
    RELOAD_SHADOW = False
    RELOAD_SHADOW_1 = False
    RELOAD_SHADOW_2 = False
    if not os.path.exists(target_shadow_logits_path) or not os.path.exists(ref_shadow_logits_path):
        RELOAD_SHADOW = True
    
    if os.path.exists(target_shadow_logits_path):
        target_shadow_logits = torch.load(records_path['target_shadow_logits'])
        if target_shadow_logits.shape[1] < shadow_numbs:
            RELOAD_SHADOW = True
            RELOAD_SHADOW_1 = True
    if os.path.exists(ref_shadow_logits_path):
        ref_shadow_logits = torch.load(records_path['ref_shadow_logits'])
        if ref_shadow_logits.shape[1] < shadow_numbs:
            RELOAD_SHADOW = True
            RELOAD_SHADOW_2 = True

    if RELOAD_SHADOW:
        print(f"Start to load shadow models, which needs a long time")
        shadow_model_list = []
        dataname = args.dataname
        if dataname == 'cifar10' or dataname == 'cinic10':
            num_classes = 10
        elif dataname == 'location':
            num_classes = 30
        elif dataname == 'cifar100' or dataname == 'texas' or dataname == 'purchase':
            num_classes = 100

        for shadow_idx in range(shadow_numbs):
            load_path = args.shadow_folder + f'{shadow_data}/shadow_model_{shadow_idx}.pth'
            print(f"Load shadow model from {load_path}")
            shadow_weight = torch.load(
                load_path, map_location=torch.device(DEVICE))
            if dataname == 'cifar10' or dataname == 'cifar100' or dataname == 'cinic10':
                shadow_model = ResNet18(num_classes=num_classes)
            elif dataname == 'location':
                shadow_model = LocationClassifier()
            elif dataname == 'texas':
                shadow_model = TexasClassifier()
            elif dataname == 'purchase':
                shadow_model = PurchaseClassifier()
            shadow_model.load_state_dict(shadow_weight)
            shadow_model.to(DEVICE)
            shadow_model.eval()
            shadow_model_list.append(shadow_model)

        if not os.path.exists(target_shadow_logits_path) or RELOAD_SHADOW_1:
            print(f"Start to inference target_shadow_logits")
            target_shadow_logits = []
            target_loader = DataLoader(target_set, batch_size=batch_size, shuffle=False, num_workers=4)
            with torch.no_grad():
                for x, y in target_loader:
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)
                    shadow_logit = torch.stack([shadow_model(x).cpu() for shadow_model in shadow_model_list], dim=1)
                    target_shadow_logits.append(shadow_logit)

            target_shadow_logits = torch.cat(target_shadow_logits)
            torch.save(target_shadow_logits, target_shadow_logits_path)
            print(f"Save target_shadow_logits_path to {target_shadow_logits_path}")
    
        if not os.path.exists(ref_shadow_logits_path) or RELOAD_SHADOW_2:
            print(f"Start to inference ref_shadow_logits")
            ref_shadow_logits = []
            ref_loader = DataLoader(ref_set, batch_size=batch_size, shuffle=False, num_workers=4)
            with torch.no_grad():
                for x, y in ref_loader:
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)
                    shadow_logit = torch.stack([shadow_model(x).cpu() for shadow_model in shadow_model_list], dim=1)
                    ref_shadow_logits.append(shadow_logit)

            ref_shadow_logits = torch.cat(ref_shadow_logits)
            torch.save(ref_shadow_logits, ref_shadow_logits_path)
            print(f"Save ref_shadow_logits with shape {ref_shadow_logits.shape} to {ref_shadow_logits_path}")


    # if not os.path.exists(target_appro_scales_path):
    #     target_ori_logits = torch.load(records_path['target_ori_logits'])
    #     target_labels = torch.load(records_path['target_labels'])
    #     target_ref_logits = torch.load(records_path['target_shadow_logits'])
        
    #     eps1 = 1e-2
    #     eps2 = 1e-5
    #     max_value =  -np.log(eps1-np.log(eps2+1))
    #     min_value = -np.log(eps1-np.log(eps2+0))
    #     target_ori_proby = model_confidence(target_ori_logits, target_labels)
    #     # target_ori_logloss = (-np.log(eps1-np.log(eps2+target_ori_proby))-min_value)/(max_value-min_value)
    #     target_ori_logloss = -np.log(eps1-np.log(eps2+target_ori_proby))

    #     shadow_target_logloss = []
    #     for ref_idx in range(args.model_numbs):
    #         shadow_signal_target = model_confidence(target_ref_logits[:,ref_idx,:].squeeze(), target_labels)
    #         # shadow_target_logloss.append((-np.log(eps1-np.log(eps2+shadow_signal_target))-min_value)/(max_value-min_value))
    #         shadow_target_logloss.append(-np.log(eps1-np.log(eps2+shadow_signal_target)))
    #     shadow_target_logloss = torch.stack(shadow_target_logloss)
        
    #     local_batch_size = 32
    #     target_loader = DataLoader(target_set, batch_size=local_batch_size, shuffle=False)
    #     scales_list_all = []
    #     above_loglossy_all, below_loglossy_all = [], []
    #     for idx, (data, label) in enumerate(target_loader):
    #         # Define 10 different scales for the noise
    #         # Define scales as 100 evenly spaced values
    #         data = data.to(DEVICE)
    #         num_samples = data.shape[0]  # Number of samples to test
    #         item = torch.arange(num_samples) + idx * local_batch_size
    #         # Adjust indices based on log loss thresholds
    #         target_threshold = target_ori_logloss[item]
    #         shadow_threshold = shadow_target_logloss[:shadow_numbs, item].mean(dim=0)
    #         # Initialize the completed mask to track samples that have met the stopping criteria
    #         completed_mask = torch.zeros(num_samples, dtype=torch.bool)  # Start with all samples as incomplete
    #         completed_search = {}  # Dictionary to store results for completed samples

    #         REPEAT=10
    #         # Initialize scales for each sample
    #         scales_list = [torch.linspace(1e-5, 5, INTER_APPRO) for _ in range(num_samples)]  # Initial scales for each sample
    #         above_loglossy = torch.zeros(num_samples)
    #         below_loglossy = torch.zeros(num_samples)
    #         for loop in range(0,10):  # Adjust the number of iterations as needed
    #             data_noisy = noise_expand(data, scales_list, REPEAT, INTER_APPRO)
    #             # Model predictions
    #             output = ori_model(data_noisy.reshape(-1, *data.shape[1:]))  # Flatten to batch dimensions
    #             labels = label.expand(INTER_APPRO, REPEAT, num_samples).flatten()
    #             proby = model_confidence(output, labels)
    #             # Calculate log loss
    #             # loglossy = (-torch.log(eps1 - torch.log(eps2 + proby.detach())) - min_value) / (max_value - min_value)
    #             loglossy = (-np.log(eps1 - np.log(eps2 + proby.detach().cpu())))
    #             loglossy = loglossy.reshape(INTER_APPRO, REPEAT, num_samples).mean(dim=1)
    #             # Skip already completed samples in this iteration
    #             active_indices = torch.arange(num_samples)[~completed_mask]
    #             active_loglossy = loglossy[:, ~completed_mask]

    #             # Create boolean masks for conditions on active samples only
    #             above_mask = (active_loglossy > target_threshold[~completed_mask]).int()
    #             below_mask = (active_loglossy < shadow_threshold[~completed_mask]).int()

    #             # Find the last "above threshold" and first "below threshold" index for each active sample
    #             above_indices = torch.argmax(above_mask.flip(dims=(0,)), dim=0)
    #             above_indices = len(scales_list[0]) - 1 - above_indices  # Flip back to correct indices
    #             below_indices = torch.argmax(below_mask, dim=0)
                
    #             # Handle cases where no index satisfies the condition
    #             above_indices[~above_mask.any(dim=0)] = 0
    #             below_indices[~below_mask.any(dim=0)] = len(scales_list[0]) - 1

    #             # Identify samples that meet the stopping criteria within active samples
    #             stop_criteria = (above_indices == below_indices) | ((above_indices == 0) & (below_indices == len(scales_list[0]) - 1)) | ((~below_mask.any(dim=0)) & (~above_mask.any(dim=0)))

    #             # # Save results for active samples meeting the criteria
    #             # for i, stop in enumerate(stop_criteria):
    #             #     if stop:
    #             #         sample_idx = int(active_indices[i])
    #             #         completed_search[sample_idx] = {
    #             #             'above_loglosy': active_loglossy[above_indices[i]].item(),
    #             #             'below_loglosy': active_loglossy[below_indices[i]].item(),
    #             #             'scale_range': (scales_list[sample_idx][above_indices[i]].item(), scales_list[sample_idx][below_indices[i]].item())
    #             #         }

    #             # Update scales for each active sample that hasn't met stopping criteria
    #             for i, sample_idx in enumerate(active_indices):
    #                 # Update scales range for the next iteration
    #                 new_scale_min = scales_list[sample_idx][above_indices[i]].item()
    #                 new_scale_max = scales_list[sample_idx][below_indices[i]].item()
    #                 scales_list[sample_idx] = torch.linspace(new_scale_min, new_scale_max, INTER_APPRO)  # New refined scales
    #                 above_loglossy[sample_idx] = active_loglossy[above_indices[i],i]
    #                 below_loglossy[sample_idx] = active_loglossy[below_indices[i],i]
    #             # Update the completed mask to include new completed samples
    #             completed_mask[active_indices[stop_criteria]] = True

    #             # Break the loop if all samples are completed
    #             if completed_mask.all():
    #                 print("All samples have met the stopping criteria.")
    #                 break
                
    #         scales_list_all += scales_list  # Initial scales for each sample
    #         above_loglossy_all.append(above_loglossy)
    #         below_loglossy_all.append(below_loglossy)
    #     scales_list_all = torch.stack(scales_list_all).cpu()
    #     above_loglossy_all = torch.cat(above_loglossy_all).cpu()
    #     below_loglossy_all = torch.cat(below_loglossy_all).cpu()
    #     torch.save(scales_list_all, target_appro_scales_path)
    #     torch.save(above_loglossy_all, target_appro_scales_path.replace('scales', 'above_loglossy'))
    #     torch.save(below_loglossy_all, target_appro_scales_path.replace('scales', 'below_loglossy'))
    #     print(f"Save scales_list_all to {target_appro_scales_path}")

    # if not os.path.exists(target_appro_logits_path):
    #     scales_list_all = torch.load(target_appro_scales_path)
    #     SIMULATE_NUM = args.model_numbs

    #     local_batch_size = 32
    #     target_loader = DataLoader(target_set, batch_size=local_batch_size, shuffle=False)
    #     shadow_target_logit_appro = []
    #     # shadow_target_label_appro = []
    #     with torch.no_grad():
    #         for idx, (data, label) in enumerate(target_loader):
    #             data = data.to(DEVICE)
    #             num_samples = data.shape[0]  # Number of samples to test
    #             item = torch.arange(num_samples) + idx * local_batch_size
    #             shadow_logit = []
    #             for shadow_idx in range(SIMULATE_NUM):
    #                 data_noisy = noise_expand(data, scales_list_all[item], 1, INTER_APPRO)
    #                 logit_noisy = ori_model(data_noisy.reshape(-1, *data.shape[1:])).cpu()
    #                 logit_noisy = logit_noisy.reshape(INTER_APPRO, num_samples, *logit_noisy.shape[1:])
    #                 shadow_logit.append(logit_noisy)
    #             # Model predictions
    #             shadow_logit = torch.stack(shadow_logit)
    #             # labels = label.expand(len(INTER_APPRO), num_samples).flatten()
    #             shadow_target_logit_appro.append(shadow_logit.detach().cpu())
    #             # shadow_target_label_appro.append(labels.cpu())

    #     shadow_target_logit_appro = torch.cat(shadow_target_logit_appro, dim=2)
    #     print(shadow_logit.shape)
    #     print(shadow_target_logit_appro.shape)
    #     # shadow_target_label_appro = torch.cat(shadow_target_label_appro)
    #     torch.save(shadow_target_logit_appro, target_appro_logits_path)
    #     print(f"Save target_appro_logits_path to {records_folder}")     

    if not os.path.exists(ref_popin_logits_path):
        ref_in_model = load_ref_in_model(args.shadow_folder + f'{dataname}/', ref_set, shadow_set, DEVICE=DEVICE)
        ref_in_model.to(DEVICE)
        ref_in_model.eval()
        ref_popin_logits = []
        ref_loader = DataLoader(ref_set, batch_size=batch_size, shuffle=False, num_workers=4)
        with torch.no_grad():
            for x, y in ref_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                ref_in_logit = ref_in_model(x)
                ref_popin_logits.append(ref_in_logit.cpu())

        ref_popin_logits = torch.cat(ref_popin_logits)
        torch.save(ref_popin_logits, ref_popin_logits_path)

        ref_logits_target = []
        target_loader = DataLoader(target_set, batch_size=batch_size, shuffle=False, num_workers=4)
        with torch.no_grad():
            for x, y in target_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                ref_in_logit = ref_in_model(x)
                ref_logits_target.append(ref_in_logit.cpu())

        ref_logits_target = torch.cat(ref_logits_target)
        torch.save(ref_logits_target, ref_popin_logits_path.replace('popin', 'targetin'))


    return records_path

def noise_expand(data, scales_list, REPEAT, INTER_APPRO):
    DEVICE = data.device
    is_binary = (len(data.unique()) == 2)
    if is_binary:
        noise = torch.stack([
            torch.bernoulli(torch.ones(*data.shape[1:], REPEAT, INTER_APPRO) * (scales_list[i] / 5))
            for i in range(data.shape[0])
        ], dim=0).to(DEVICE)  # 形状为 (num_samples, ..., REPEAT, INTER_APPRO)
        if len(data.shape)==4:
            noise = noise.permute(-1, -2, 0,1,2,3)
        elif len(data.shape)==3:
            noise = noise.permute(-1, -2, 0,1,2)
        elif len(data.shape)==2:
            noise = noise.permute(-1, -2, 0,1)
        data_noisy = data[None,].expand_as(noise)
        data_noisy = torch.bitwise_xor(data_noisy.int(), noise.int()).float()
    else:
        noise = torch.stack([
            torch.randn(*data.shape[1:], REPEAT, INTER_APPRO) * scales_list[i]
            for i in range(data.shape[0])
        ], dim=0).to(DEVICE)   # Shape (num_scales, num_samples, ...)
        if len(data.shape)==4:
            noise = noise.permute(-1, -2, 0,1,2,3)
        elif len(data.shape)==3:
            noise = noise.permute(-1, -2, 0,1,2)
        elif len(data.shape)==2:
            noise = noise.permute(-1, -2, 0,1)
        data_noisy = data[None,] + noise  # Broadcasting across scales and samples
    
    return data_noisy

def analysis_mat(target_unlearn_flags, target_unl_proby, score, threshold=0.5):

    interidx = set(np.where((target_unl_proby<threshold))[0].tolist()) & set(np.where(score<threshold)[0].tolist()) & set(np.where(target_unlearn_flags==1)[0].tolist())
    interidx = list(interidx)
    print('unlearned | low proby,  but low score  | generalize bad, but shadow generalize worse than target |', len(interidx), interidx[:3])

    interidx = set(np.where((target_unl_proby>threshold))[0].tolist()) & set(np.where(score>threshold)[0].tolist()) & set(np.where(target_unlearn_flags==1)[0].tolist())
    interidx = list(interidx)
    print('unlearned | high proby, but high score | generalize well, and shadow approximate target, we hope |', len(interidx), interidx[:3])

    interidx = set(np.where((target_unl_proby<threshold))[0].tolist()) & set(np.where(score>threshold)[0].tolist()) & set(np.where(target_unlearn_flags==1)[0].tolist())
    interidx = list(interidx)
    print('unlearned | low proby,  but high score | generalize bad, and shadow approximate target, we hope  |', len(interidx), interidx[:3])

    interidx = set(np.where((target_unl_proby>threshold))[0].tolist()) & set(np.where(score<threshold)[0].tolist()) & set(np.where(target_unlearn_flags==1)[0].tolist())
    interidx = list(interidx)
    print('*unlearned| high proby, but low score  | generalize well, but shadow generalize worse than target|', len(interidx), interidx[:3], '| ***hard to detect***')

    print('----------------------')
    interidx = set(np.where((target_unl_proby<threshold))[0].tolist()) & set(np.where(score<threshold)[0].tolist()) & set(np.where(target_unlearn_flags==0)[0].tolist())
    interidx = list(interidx)
    print('retained  | low proby,  but low score  | fit bad, shadow generalize worse than target fit, we hope  |', len(interidx), interidx[:3])

    interidx = set(np.where((target_unl_proby>threshold))[0].tolist()) & set(np.where(score>threshold)[0].tolist()) & set(np.where(target_unlearn_flags==0)[0].tolist())
    interidx = list(interidx)
    print('retained  | high proby, but high score | fit well, shadow generalize better than target fit         |', len(interidx), interidx[:3], '| ***hard to detect***')

    interidx = set(np.where((target_unl_proby<threshold))[0].tolist()) & set(np.where(score>threshold)[0].tolist()) & set(np.where(target_unlearn_flags==0)[0].tolist())
    interidx = list(interidx)
    print('*retained | low proby,  but high score | fit bad, shadow generalize better than target fit          |', len(interidx), interidx[:3])

    interidx = set(np.where((target_unl_proby>threshold))[0].tolist()) & set(np.where(score<threshold)[0].tolist()) & set(np.where(target_unlearn_flags==0)[0].tolist())
    interidx = list(interidx)
    print('retained  | high proby, but low score  | fit well, but shadow generalize worse than target, we hope |', len(interidx), interidx[:3])
    print('----------------------\n')

def fetch_data_model(args,  verbose=False, SEED = 42, DEVICE='cpu'):
    acc_values = {'training': 0, 'testing': 0, 'forget': 0, 'retain': 0}
    RNG = torch.Generator()    
    RNG.manual_seed(SEED)  # Initialize RNG with seed 42 for dataset partition, whether newly created or provided
    
    dataname = args.dataname
    unlearn_method = args.unlearn_method
    unlearn_type = args.unlearn_type
    if unlearn_type == 'set_random':
        forget_class = None
        forget_size = args.forget_size
    elif unlearn_type == 'one_class':
        forget_class = args.forget_class
        forget_size = 1
    elif unlearn_type == 'class_percentage':
        forget_class = args.forget_class
        forget_size = args.forget_class_ratio

    EXACT_FLAG = False if unlearn_method != 'retrain' else args.EXACT_FLAG
    
    SUFFIX = args.SUFFIX

    if dataname == 'cifar10' or dataname == 'cinic10':
        num_classes = 10
    elif dataname == 'cifar100' or dataname == 'texas' or dataname == 'purchase':
        num_classes = 100
    elif dataname == 'location':
        num_classes = 30
    
    # split dataname
    init_train_loader, retain_loader, forget_loader, val_loader, test_loader, shadow_set, cut_train_set, unlearn_flags, val_set  = get_data_loaders(dataname, unlearn_type=unlearn_type, forget_class=forget_class, forget_size=forget_size, SEED=SEED)
    # prepare shadowmodels
    shadow_path = f'LIRA_checkpoints/shadow_models/{dataname}'
    shadow_training(shadow_set, dataname, ratio=0.8, ORIGIN=True, shadow_nums=args.model_numbs, shadow_path=shadow_path, DEVICE=DEVICE, VERBOSE=verbose)

    if dataname == 'cifar10' or dataname == 'cifar100'  or dataname == 'cinic10':
        weights_pretrained, _ = load_pretrain_weights(DEVICE, TRAIN_FROM_SCRATCH=True, RETRAIN=False, dataname=dataname,
                                                train_loader=init_train_loader, test_loader=val_loader, checkpoints_folder='LIRA_checkpoints', SEED=SEED)
        # load model with pre-trained weights
        model = ResNet18(num_classes=num_classes)
        model.load_state_dict(weights_pretrained)
        model.to(DEVICE)
        model.eval()
    else:
        model = train_classifier(init_train_loader, dataname, val_loader, 'ori', DEVICE, checkpoints_folder='LIRA_checkpoints', SEED=SEED)
        model.to(DEVICE)
        model.eval()
    # [optional] pretrained model accuracy
    if verbose:
        acc_values['training'] = 100.0 * accuracy(model, init_train_loader, DEVICE)
        acc_values['testing'] = 100.0 * accuracy(model, test_loader, DEVICE)
        print(
            f"Train set accuracy: {acc_values['training'] :0.1f}%")
        print(
            f"Test set accuracy: {acc_values['testing']:0.1f}%")

    # Here unlearning model should be fetched from a function
    if unlearn_method == 'retrain':
        if dataname == 'cifar10' or dataname == 'cifar100' or dataname == 'cinic10':
            weights_rt_pretrained, _ = load_pretrain_weights(DEVICE, TRAIN_FROM_SCRATCH=True, RETRAIN=True, dataname=dataname,
                                                            train_loader=retain_loader, test_loader=val_loader, checkpoints_folder='LIRA_checkpoints', SUFFIX=SUFFIX, SEED=SEED)
            ul_model = ResNet18(num_classes=num_classes)
            ul_model.load_state_dict(weights_rt_pretrained)
            ul_model.to(DEVICE)
            ul_model.eval()
        else:
            ul_model = train_classifier(retain_loader, dataname, val_loader, 'retrain', DEVICE, checkpoints_folder='LIRA_checkpoints', SUFFIX=SUFFIX, SEED=SEED)
            ul_model.to(DEVICE)
            ul_model.eval()
    else:
        if dataname == 'cifar10' or dataname == 'cifar100' or dataname == 'cinic10':
            ul_model = ResNet18(num_classes=num_classes)
            ul_model.load_state_dict(weights_pretrained)
            ul_model.to(DEVICE)
            ul_model = get_unlearn_model(ul_model, dataname, args.unlearn_method, retain_loader, forget_loader, val_loader, test_loader, num_classes, DEVICE)
        else:
            ul_model = train_classifier(retain_loader, dataname, val_loader, 'unlearn', DEVICE, checkpoints_folder='LIRA_checkpoints', SUFFIX=SUFFIX, SEED=SEED)
            ul_model.to(DEVICE)
            ul_model = get_unlearn_model(ul_model, dataname, args.unlearn_method, retain_loader, forget_loader, val_loader, test_loader, num_classes, DEVICE)

    # [optional] pretrained retrain model accuracy
    # print its accuracy on retain and forget set
    if verbose:
        acc_values['retain'] = 100.0 * accuracy(ul_model, retain_loader, DEVICE)
        acc_values['forget'] = 100.0 * accuracy(ul_model, forget_loader, DEVICE)
        print(
            f"Retain set accuracy: {acc_values['retain'] :0.1f}%")
        print(
            f"Forget set accuracy: {acc_values['forget']:0.1f}%")
    
    return model, ul_model, cut_train_set, unlearn_flags, val_set, test_loader, shadow_path, shadow_set, SUFFIX, acc_values

def create_shadow_lists(total_models=128, model_nums=1):
    """
    Create shadow model lists by dividing total models into equal groups.
    
    Args:
        total_models (int): Total number of shadow models (default: 128)
        model_nums (int): Number of groups to divide into (default: 1)
    
    Returns:
        list: List of shadow model groups
    """
    # Calculate models per group
    models_per_group = model_nums
    group_nums = min(total_models // model_nums, 10)
    
    # Create the groups using list comprehension
    shadow_lists = [
        list(range(i * models_per_group, (i + 1) * models_per_group))
        for i in range(group_nums)
    ]
    
    return shadow_lists


def get_scores(args_input):
    args = copy.deepcopy(args_input)
    # args.SHADOW_IDXs = random.sample(range(128), args.model_numbs)
    args.SHADOW_IDXs = list(range(args.model_numbs))
    if args.SHADOW_AVE_FLAG:
        shadow_lists = create_shadow_lists(total_models=128, model_nums=args.model_numbs)    
    else:
        shadow_lists = [list(range(args.model_numbs))]  
    args.SUFFIX = args.dataname + '_' + args.unlearn_method + '_' + args.unlearn_type + '_' + str(args.forget_size)
    SEED_init = args.SEED_init
    RNG = torch.Generator()
    RNG.manual_seed(SEED_init)
    Evaluation = []
    print(f"shadow_lists: {shadow_lists}")  
    ALL_Scores = []
    ALL_unlearn_flags = []
    for idx,shadow_list in enumerate(shadow_lists):
        Scores = {}
        Unlearn_flags = []
        Unlearn_proby = []
        print(f"{idx}-th shadow_list: {shadow_list}")
        args.SHADOW_IDXs = shadow_list
        for loop in range(args.LOOP):
            if args.unlearn_type == 'set_random':
                SEED = SEED_init + loop       
                print(f"Loop {loop}, SEED {SEED}")   
            else:
                SEED = SEED_init  
                args.forget_class = args.CLASS_init+loop  
                print(f"Loop {loop}, SEED {SEED_init}, CLASS {args.forget_class}, TYPE {args.unlearn_type}")         
                if args.unlearn_type == 'one_class':
                    args.SUFFIX = args.dataname + '_' + args.unlearn_method + '_' + args.unlearn_type + '_' + str(args.forget_class)
                elif args.unlearn_type == 'class_percentage':
                    args.SUFFIX = args.dataname + '_' + args.unlearn_method + '_' + args.unlearn_type + '_' + 'class' + str(args.forget_class) + '_' + str(args.forget_class_ratio)
            
            audit_predicts, target_unlearn_flags, target_unl_proby = audit_nonmem_mono(args, DEVICE=DEVICE, SEED=SEED)                                   
            for key in audit_predicts.keys():
                if key not in Scores:
                    Scores[key] = []
                Scores[key].append(audit_predicts[key])

            Unlearn_flags.append(target_unlearn_flags)
            Unlearn_proby.append(target_unl_proby)


        for key in Scores.keys():
            Scores[key] = np.stack(Scores[key], axis=1)

        Unlearn_flags = np.stack(Unlearn_flags, axis=1)

        Unlearn_flags = Unlearn_flags.flatten()
        Unlearn_flags = Unlearn_flags
        Unlearn_proby = torch.concatenate(Unlearn_proby, dim=0)

        for key in Scores.keys():
            Scores[key] = Scores[key].flatten()
        
        Evaluation_local = {}
        for key in Scores.keys():
            # tpr@lowfpr==0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1
            # tnr@lowfnr==0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1
            auc_score, acc_score, desired_fpr, desired_tpr, desired_tnr = auc_extended(
                    1-Unlearn_flags, 1-Scores[key], verbose=False)
            
            # print(f'{key}')
            # analysis_mat(Unlearn_flags, Unlearn_proby, Scores[key])
            
            if key not in Evaluation_local.keys():
                Evaluation_local[key] = {}

            stat_cross_sample =[Scores[key][Unlearn_flags==1].mean()
                                , Scores[key][Unlearn_flags==1].std()
                                , Scores[key][Unlearn_flags==0].mean()
                                , Scores[key][Unlearn_flags==0].std()]

            Evaluation_local[key] = np.concatenate(
                [[auc_score], [acc_score], [desired_tpr[0],desired_tpr[1],desired_tpr[4]], [desired_tnr[0],desired_tnr[1],desired_tnr[4]],  stat_cross_sample])
        
        Evaluation.append(Evaluation_local)
        ALL_Scores.append(Scores)
        ALL_unlearn_flags.append(Unlearn_flags)

    Evaluation_all = {}
    Evaluation_all_var = {}
    for key in Evaluation[0].keys():
        Evaluation_all[key] = np.stack([Evaluation[i][key] for i in range(len(Evaluation))], axis=0)
        # calculate mean
        Evaluation_all_var[key] = Evaluation_all[key].std(axis=0)
        Evaluation_all[key] = Evaluation_all[key].mean(axis=0)
    

    print('\nstart_print')
    print(args)
    print( Scores.keys())

    # print the mean and std of the scores with keys
    print(f'{"Metric values ":<30}', f'{"auc":<6}', f'{"acc":<6}',
        f'{"tpr0.0":<6}', f'{".-5":<6}', f'{".-2":<6}', 
        f'{"tnr0.0":<6}', f'{".-5":<6}', f'{".-2":<6}', 
        f'{"1-mean":<6}', f'{"1-std":<6}', f'{"0-mean":<6}', f'{"0-std":<6}',
        f'{"cm-std1":<6}', f'{"cm-std0":<6}')
    for key in Evaluation_all.keys():
        if 'p_LiRA' in key:
            print(
                f'{key:<30}', 
                " ".join(f"{x:.2e}" if i >= len(Evaluation_all[key]) - 6 else f"{x:.4f}" 
                        for i, x in enumerate(Evaluation_all[key]))
            )
            print(
                f'{key + "_var":<30}',
                " ".join(f"{x:.2e}" if i >= len(Evaluation_all_var[key]) - 6 else f"{x:.4f}" 
                        for i, x in enumerate(Evaluation_all_var[key]))
            )
        else:
            print(f'{key:<30}', " ".join(f"{x:.4f}" for x in Evaluation_all[key]))
            print(f'{key + "_var":<30}', " ".join(f"{x:.4f}" for x in Evaluation_all_var[key]))

    print('end_print\n\n')

    records_folder = args.records_folder + f'seed_{SEED_init}/'
    if not os.path.exists(records_folder):
        os.makedirs(records_folder)

    save_name = records_folder + f'shift_{args.SUFFIX}_scores_{SEED_init}_{args.LOOP}_ref{args.model_numbs}.pth'
    torch.save(ALL_Scores, save_name)
    save_name = records_folder + f'shift_{args.SUFFIX}_unlearn_flags_{SEED_init}_{args.LOOP}_ref{args.model_numbs}.pth'
    torch.save(ALL_unlearn_flags, save_name)
 


    # calculate AUC

    
def parse_args():
    parser = argparse.ArgumentParser(description="Test UnleScore on various datasets and unlearn methods.")
    
    parser.add_argument('--dataname', type=str, default=None, choices=['cifar10', 'cifar100', 'cinic10', 'purchase'],
                        help="Dataset to use")
    parser.add_argument('--unlearn_method', type=str, default='retrain', 
                        choices=['retrain', 'finetune', 'ssd', 'fisher', 'forsaken', 'l_codec', 'ascent', 'boundary_expanding', 'all'],
                        help="Unlearn method to apply")
    parser.add_argument('--unlearn_type', type=str, default='set_random', 
                        choices=['one_class', 'class_percentage', 'set_random'],
                        help="Type of unlearning process")
    parser.add_argument('--forget_class', type=int, default=0, 
                        help="Class index to forget (only applicable for 'one_class' and 'class_percentage')")
    parser.add_argument('--forget_size', type=int, default=500, 
                        help="Number of instances to forget (only applicable for 'set_random')")
    parser.add_argument('--forget_class_ratio', type=float, default=0.1, 
                        help="Ratio of class to forget (only applicable for 'class_percentage')")
    parser.add_argument('--model_numbs', type=int, default=1, 
                        help="Number of models to train")
    parser.add_argument('--SHADOW_AVE_FLAG', action='store_true', # default=False, if sepecified using '''--SHADOW_AVE_FLAG''', then it is True
                        help="Whether to average shadow models")
    parser.add_argument('--LOOP', type=int, default=10, 
                        help="Number of training loops")
    parser.add_argument('--EXACT_FLAG', action='store_true', 
                        help="Whether to use exact flag during unlearning")
    parser.add_argument('--shadow_folder',type=str, default='LIRA_checkpoints/shadow_models/')
    parser.add_argument('--records_folder',type=str, default='LIRA_checkpoints/records/')
    parser.add_argument('--SEED_init', type=int, default=42)
    parser.add_argument('--CLASS_init', type=int, default=0)
    parser.add_argument('--shift_type', type=str, default='shadow_data', 
                        choices=['shadow_data', 'model_arch'])
    parser.add_argument('--INTERAPPROX', type=int, default=100)
    parser.add_argument('--metrics', nargs='+', default=['p_Unleak', 'p_LiRA', 'p_update_LiRA', 'p_EMIA', 'p_EMIA_p', 'p_RMIA', 'p_SimpleDiff', 'p_LDiff', 'p_interapprox', 'p_random'],
                        help="Metrics to evaluate")
    parser.add_argument('--MODEL_CHECK', action='store_true', 
                        help="Whether to use exact flag during unlearning")

    
    return parser.parse_args()

# main function
if __name__ == "__main__":
    args_input = parse_args()

    if args_input.shift_type == 'shadow_data':
        for dataname in ['cinic10','cifar10']:
        # for dataname in ['cifar100']:
            args_input.dataname = dataname
            get_scores(args_input)

        # args.update({'unlearn_type': 'class_percentage', 'forget_class_ratio': 0.5})
        # audit_scores, audit_scores_rt, nonmem_records, forget_records, nonmem_records_rt, retain_records = get_scores(args)
        # args.update({'unlearn_type': 'one_class'})
        # audit_scores, audit_scores_rt, nonmem_records, forget_records, nonmem_records_rt, retain_records = get_scores(args)