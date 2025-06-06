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



def audit_nonmem_mono(args, DEVICE=torch.device("cpu"), breakcount=0, batch_size=1024):
    
    # Check if all logits are saved, if not, save them
    records_folder = args.records_folder + f'Breaks{args.BREAKs}/breakcount_{breakcount}/'
    records_path = load_records(records_folder, args, batch_size=batch_size, DEVICE=DEVICE, breakcount=breakcount)
    print(records_path)

    # for metrics in args.metrics:
    #     if metrics not in ['p_Unleak', 'p_LiRA', 'p_update_LiRA', 'p_EMIA', 'p_RMIA', 'p_SimpleDiff', 'p_Dratio']:
    #         raise ValueError(f"Metrics {metrics} is not supported")

    audit_predicts = {}
    if 'p_Unleak' in args.metrics:
        # train attack models on nonmem set
        attack_model = train_attack_model(
            args, model_list=args.SHADOW_IDXs, shadow_path= args.shadow_folder + f'{args.dataname}', DEVICE=DEVICE)
        
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

def load_records(records_folder, args, batch_size=1024, DEVICE='cpu', breakcount=0):

    if not os.path.exists(records_folder):
        os.makedirs(records_folder)
    if not os.path.exists(args.records_folder + 'share_ori/'):
        os.makedirs(args.records_folder + 'share_ori/')
    shadow_numbs = 128

    dataname = args.dataname
    unlearn_method = args.unlearn_method
    unlearn_type = args.unlearn_type
    
    records_seed_folder = args.records_folder + f'seed_{args.SEED_init}/'

    target_ori_signal_off_suffix = f'{dataname}_off_fitting_{len(args.SHADOW_IDXs)}_'+'_'.join(map(str, args.SHADOW_IDXs))+'.pth'
    if len(target_ori_signal_off_suffix) > 200:
        target_ori_signal_off_suffix = target_ori_signal_off_suffix.split('.pkl')[0][:200]+'.pth'

    SUFFIX = args.SUFFIX
    print(SUFFIX)
    target_unl_logits_path = records_folder + f'{SUFFIX}_target_{args.select_idx}_unl_logits.pth' # #number x #class
    target_unlearn_flags_path = records_folder + f'{SUFFIX}_target_unlearn_flags.pth'  # #number
    ref_unl_logits_path = records_folder + f'{SUFFIX}_ref_{args.select_idx}_unl_logits.pth' # #number x #class
    accuracy_path = records_folder + f'{SUFFIX}_accuracy.pth' # #number x #ref x #class

    target_ori_logits_path = args.records_folder + f'share_ori/{dataname}_target_ori_logits.pth' # #number x #class
    target_ori_signal_off_path = args.records_folder + f'share_ori/' + target_ori_signal_off_suffix # #number x #class
    target_labels_path = args.records_folder + f'share_ori/{dataname}_target_labels.pth'  # #number
    target_shadow_logits_path = args.records_folder + f'share_ori/{dataname}_target_shadows_logits.pth' # #number x #shadow x #class
    ref_ori_logits_path = args.records_folder  + f'share_ori/{dataname}_ref_ori_logits.pth' # #number x #class
    ref_labels_path = args.records_folder + f'share_ori/{dataname}_ref_labels.pth' # #number x #class
    ref_shadow_logits_path = args.records_folder + f'share_ori/{dataname}_ref_shadow_logits.pth' # #number x #ref x #class
    ref_popin_logits_path = args.records_folder + f'share_ori/{dataname}_ref_popin_logits.pth' # #number x #class

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
    
    ori_model, unl_model, target_set, unlearn_flags, ref_set, test_loader, shadow_path, shadow_set, SUFFIX, accuracy = fetch_data_model(args, verbose=True, breakcount=breakcount, SEED=args.SEED_init)

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
        eps1 = 1e-2 if args.dataname not in ['purchase', 'texas'] else 1e-1
        eps2 = 1e-5 if args.dataname not in ['purchase', 'texas'] else 1e-6
        max_value =  -np.log(eps1-np.log(eps2+1))
        min_value = -np.log(eps1-np.log(eps2+0))

        shadow_fit = 0
        shadow_path= args.shadow_folder + f'{args.dataname}'
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
        shadow_model_list = []
        dataname = args.dataname
        if dataname == 'cifar10' or dataname == 'cinic10':
            num_classes = 10
        elif dataname == 'location':
            num_classes = 30
        elif dataname == 'cifar100' or dataname == 'texas' or dataname == 'purchase':
            num_classes = 100

        for shadow_idx in range(shadow_numbs):
            load_path = args.shadow_folder + f'{dataname}/shadow_model_{shadow_idx}.pth'
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
            print(f"Save target_shadow_logits_path to {records_folder}")
        
        if not os.path.exists(ref_shadow_logits_path) or RELOAD_SHADOW_2:
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
            print(f"Save ref_shadow_logits_path to {records_folder}")
    
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

        ref_targetin_logits = []
        target_loader = DataLoader(target_set, batch_size=batch_size, shuffle=False, num_workers=4)
        with torch.no_grad():
            for x, y in target_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                ref_in_logit = ref_in_model(x)
                ref_targetin_logits.append(ref_in_logit.cpu())

        ref_targetin_logits = torch.cat(ref_targetin_logits)
        torch.save(ref_targetin_logits, ref_popin_logits_path.replace('popin', 'targetin'))


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

def robust_normalize(scores, percentile_range=(1, 99)):
    """
    Normalize scores to [0,1] with outlier handling.
    
    Args:
        scores: numpy array of scores
        percentile_range: tuple of (lower, upper) percentiles to clip
    
    Returns:
        normalized scores in range [0,1]
    """
    # Make a copy to avoid modifying original data
    scores_copy = scores.copy()
    
    # Get percentile values for clipping
    lower = np.percentile(scores_copy, percentile_range[0])
    upper = np.percentile(scores_copy, percentile_range[1])
    
    # Clip values to remove extremes
    scores_clipped = np.clip(scores_copy, lower, upper)
    
    # Min-max normalization
    normalized = (scores_clipped - scores_clipped.min()) / \
                (scores_clipped.max() - scores_clipped.min())
    
    return normalized

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

def fetch_data_model(args,  verbose=False, breakcount=0, SEED = 42):
    acc_values = {'training': 0, 'testing': 0, 'forget': 0, 'retain': 0}
    RNG = torch.Generator()    
    RNG.manual_seed(SEED)  # Initialize RNG with seed 42 for dataset partition, whether newly created or provided
    
    dataname = args.dataname
    unlearn_method = args.unlearn_method
    unlearn_type = args.unlearn_type
    forget_class = args.forget_class
    forget_size = args.forget_size
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
    if dataname == 'cifar10' or dataname == 'cifar100'  or dataname == 'cinic10':
        weights_pretrained, _ = load_pretrain_weights_local(DEVICE, TRAIN_FROM_SCRATCH=True, RETRAIN=False, dataname=dataname,
                                                train_loader=init_train_loader, test_loader=val_loader, checkpoints_folder='LIRA_checkpoints', 
                                                SEED=SEED, BREAKs=args.BREAKs, breakcount=breakcount,
                                                shadow_folder=args.shadow_folder, shadow_idx=args.select_idx)
        # load model with pre-trained weights
        internal_model = ResNet18(num_classes=num_classes)
        internal_model.load_state_dict(weights_pretrained)
        internal_model.to(DEVICE)
        internal_model.eval()
    else:
        internal_model = train_classifier_local(init_train_loader, dataname, val_loader, 'ori', DEVICE, 
                                                checkpoints_folder='LIRA_checkpoints', SEED=SEED, BREAKs=args.BREAKs, breakcount=breakcount,
                                                shadow_folder=args.shadow_folder, shadow_idx=args.select_idx)
        internal_model.to(DEVICE)
        internal_model.eval()
    # [optional] pretrained internal_model accuracy
    if verbose:
        acc_values['training'] = 100.0 * accuracy(internal_model, init_train_loader, DEVICE)
        acc_values['testing'] = 100.0 * accuracy(internal_model, test_loader, DEVICE)
        print(
            f"Train set accuracy: {acc_values['training'] :0.1f}%")
        print(
            f"Test set accuracy: {acc_values['testing']:0.1f}%")
        
    
    return model, internal_model, cut_train_set, unlearn_flags, val_set, test_loader, shadow_path, shadow_set, SUFFIX, acc_values


def create_shadow_lists(total_models=128, model_nums=1, min_group_size=10):
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
    group_nums = min(total_models // model_nums, min_group_size)
    
    # Create the groups using list comprehension
    shadow_lists = [
        list(range(i * models_per_group, (i + 1) * models_per_group))
        for i in range(group_nums)
    ]
    
    return shadow_lists

def kl_div_score(y_true, y_pred):
    hist_true, bins = np.histogram(y_true, bins=20, density=True)
    hist_pred, _ = np.histogram(y_pred, bins=bins, density=True)
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    kl_div = np.sum(hist_true * np.log((hist_true + eps)/(hist_pred + eps)))
    return kl_div

def get_scores(args_input):
    args = copy.deepcopy(args_input)
    print(args)
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
    mse_value_split_lists = {}
    pearsonr_value_split_lists = {}
    spearman_value_split_lists = {}
    kl_div_value_split_lists = {}
    records_folder = args.records_folder + f'Breaks{args.BREAKs}/'
    for idx,shadow_list in enumerate(shadow_lists):
        Scores = {}
        Unlearn_flags = []
        Unlearn_proby = []
        print(f"{idx}-th shadow_list: {shadow_list}")
        args.SHADOW_IDXs = shadow_list
        mse_value_split_loop = {}
        pearsonr_value_split_loop = {}
        spearman_value_split_loop = {}
        kl_div_value_split_loop = {}
        for loop in range(args.LOOP):
            if args.unlearn_type == 'set_random':
                args.SUFFIX = args.dataname + '_' + args.unlearn_method + '_' + args.unlearn_type + '_' + str(args.forget_size)
                SEED_init = args.SEED_init

            args.select_idx = loop+args.init_idx
            Member_scores = []
            Scores = {}
            save_name = records_folder + f'{args.SUFFIX}_internal_shadow_{args.select_idx}_ref_{args.model_numbs}'+'_'.join(map(str, shadow_list))+'.pth'
            if len(save_name) > 200:
                save_name = save_name.split('.pth')[0][:200]+'.pth'
            # save_name = records_folder + f'{args.SUFFIX}_internal_shadow_{args.select_idx}_ref_{args.model_numbs}.pth'

            # if os.path.exists(save_name):
            #     Scores = torch.load(save_name)
            #     print(f"Load {save_name}")
            #     for key in Scores.keys():
            #         break
            #     for breakcount in range(args.BREAKs+1):
            #         target_member_score = (breakcount/args.BREAKs)*np.ones_like(Scores[key][:,0])
            #         print(f'mean of target_member_score breakcount:{breakcount}: {np.mean(target_member_score)}')
            #         Member_scores.append(target_member_score)
            #     Member_scores = np.stack(Member_scores, axis=1)

            # else:
            for breakcount in range(args.BREAKs+1):
                print(f"loop {loop} BREAKs {args.BREAKs}, breakcount {breakcount}")         
                audit_predicts, target_member_score, target_unl_proby = audit_nonmem_mono(args, DEVICE=DEVICE, breakcount=breakcount)
                target_member_score = (breakcount/args.BREAKs)*torch.ones_like(target_member_score)

                for key in audit_predicts.keys():
                    if key not in Scores:
                        Scores[key] = []
                    Scores[key].append(1-audit_predicts[key])

                Member_scores.append(target_member_score)

            for key in Scores.keys():
                Scores[key] = np.stack(Scores[key], axis=1)

            Member_scores = np.stack(Member_scores, axis=1)
            # save scores and predictions
            if not os.path.exists(records_folder):
                os.makedirs(records_folder)
            torch.save(Scores, save_name)
            # have to process the Member_scores to avoid the case that when predicting the same value, the mse is very small


            mse_value_split_keys = {}
            pearsonr_value_split_keys = {}
            spearman_value_split_keys = {}
            kl_div_value_split_keys = {}

            for key in Scores.keys():
                # normalize the scores to [0, 1]
                # before normalize, try to avoid the extreme case so that the most of the values are centered in the middle
                Scores[key] = (Scores[key] - Scores[key].min()) / (Scores[key].max() - Scores[key].min())
                # Scores[key] = robust_normalize(Scores[key])
                # save the histogram of the scores in to a figure
                # fig = plt.figure()
                # plt.hist(Scores[key].reshape(-1), bins=100)
                # plt.title(f'{key} scores')
                # plt.xlabel('scores')
                # plt.ylabel('counts')
                # plt.savefig(f'tmp_figures/0_{args.SUFFIX}_{key}_hist.png')

                mse_value_split = []
                mse_value_split.append(np.mean((Scores[key] - Member_scores)**2))
                for b in range(Member_scores.shape[1]):
                    mse_value_split.append(np.mean((Scores[key][:,b] - Member_scores[:,b])**2))
                mse_value_split_keys[key] = mse_value_split

                kl_div_value_split = []
                kl_div_value_split.append(kl_div_score(Member_scores.reshape(-1), Scores[key].reshape(-1)))
                kl_div_value_split_keys[key] = kl_div_value_split
                # calculate the NDCG value for each key, the prediciton is Scores[key], the ground truth is Member_scores
                pearsonr_value_split_keys[key] = stats.pearsonr(Member_scores.reshape(-1), Scores[key].reshape(-1)).correlation
                spearman_value_split_keys[key] = stats.spearmanr(Member_scores.reshape(-1), Scores[key].reshape(-1)).correlation

            mse_value_split_loop[loop] = mse_value_split_keys
            kl_div_value_split_loop[loop] = kl_div_value_split_keys
            pearsonr_value_split_loop[loop] = pearsonr_value_split_keys
            spearman_value_split_loop[loop] = spearman_value_split_keys

        # calculate the mean Mse_value_split_all over all loops
        mse_value_split_all = {}
        pearsonr_value_split_all = {}
        spearman_value_split_all = {}
        kl_div_value_split_all = {}
        for key in mse_value_split_loop[0].keys():
            mse_value_split_all[key] = np.mean([mse_value_split_loop[loop][key] for loop in range(args.LOOP)], axis=0)
            kl_div_value_split_all[key] = np.mean([kl_div_value_split_loop[loop][key] for loop in range(args.LOOP)], axis=0)
            pearsonr_value_split_all[key] = np.mean([pearsonr_value_split_loop[loop][key] for loop in range(args.LOOP)], axis=0)
            spearman_value_split_all[key] = np.mean([spearman_value_split_loop[loop][key] for loop in range(args.LOOP)], axis=0)
        mse_value_split_lists[idx] = mse_value_split_all
        kl_div_value_split_lists[idx] = kl_div_value_split_all
        pearsonr_value_split_lists[idx] = pearsonr_value_split_all
        spearman_value_split_lists[idx] = spearman_value_split_all

    mse_value_split_all = {}
    pearsonr_value_split_all = {}
    spearman_value_split_all = {}
    kl_div_value_split_all = {}

    mse_value_split_all_vars = {}
    pearsonr_value_split_all_vars = {}
    spearman_value_split_all_vars = {}
    kl_div_value_split_all_vars = {}

    for key in mse_value_split_lists[0].keys():
        mse_value_split_all_vars[key] = np.std([mse_value_split_lists[idx][key] for idx in range(len(shadow_lists))], axis=0)
        pearsonr_value_split_all_vars[key] = np.std([pearsonr_value_split_lists[idx][key] for idx in range(len(shadow_lists))], axis=0)
        spearman_value_split_all_vars[key] = np.std([spearman_value_split_lists[idx][key] for idx in range(len(shadow_lists))], axis=0)
        kl_div_value_split_all_vars[key] = np.std([kl_div_value_split_lists[idx][key] for idx in range(len(shadow_lists))], axis=0)

        mse_value_split_all[key] = np.mean([mse_value_split_lists[idx][key] for idx in range(len(shadow_lists))], axis=0)
        pearsonr_value_split_all[key] = np.mean([pearsonr_value_split_lists[idx][key] for idx in range(len(shadow_lists))], axis=0)
        spearman_value_split_all[key] = np.mean([spearman_value_split_lists[idx][key] for idx in range(len(shadow_lists))], axis=0)
        kl_div_value_split_all[key] = np.mean([kl_div_value_split_lists[idx][key] for idx in range(len(shadow_lists))], axis=0)

    print('\nstart_print')
    print(args)
    print( Scores.keys())
    # print the mean and std of the scores with keys
    print(f'{"Metric values ":<30}',
        f'{"all":<6}',
        f'{"b0":<6}', f'{"b1":<6}',
        f'{"b2":<6}', f'{"b3":<6}', 
        f'{"b4":<6}', f'{"b5":<6}', 
        f'{"b6":<6}', f'{"b7":<6}', 
        f'{"b8":<6}', f'{"b9":<6}',
        f'{"b10":<6}', f'{"b11":<6}',
        f'{"b12":<6}', f'{"b13":<6}',
        f'{"b14":<6}', f'{"b15":<6}',
        f'{"b16":<6}', f'{"b17":<6}',
        f'{"b18":<6}', f'{"b19":<6}',
        )

    random_guess = np.ones_like(Scores[key]) * 0.5
    random_guess_b_mse = []
    random_guess_b_mse.append(np.mean((random_guess - Member_scores)**2))
    for b in range(Member_scores.shape[1]):
        random_guess_b_mse.append(np.mean((random_guess[:,b] - Member_scores[:,b])**2))
    random_guess_ndcg = stats.pearsonr(Member_scores.reshape(-1), random_guess.reshape(-1)).correlation
    random_guess_spearman = stats.spearmanr(Member_scores.reshape(-1), random_guess.reshape(-1)).correlation
    random_guess_kl_div = []
    random_guess_kl_div.append(kl_div_score(Member_scores.reshape(-1), random_guess.reshape(-1)))
    
    # print(f'{'random_guess':<24}', f'{"mse":<6}',  " ".join(f"{x:.4f}" for x in random_guess_b_mse))
    # print(f'{'random_guess':<24}', f'{"kl_div":<6}', " ".join(f"{x:.4f}" for x in random_guess_kl_div))
    # print(f'{'random_guess':<24}', f'{"NDCG":<6}', "".join(f"{random_guess_ndcg:.4f}"))
    # print(f'{'random_guess':<24}', f'{"spearman":<6}', "".join(f"{random_guess_spearman:.4f}"))


    for key in mse_value_split_all.keys():
        print(f'{key:<24}', f'{"mse":<6}',  " ".join(f"{x:.4f}" for x in mse_value_split_all[key]))
        print(f'{key+ "_var":<24}', f'{"mse":<6}',  " ".join(f"{x:.4f}" for x in mse_value_split_all_vars[key]))
    for key in mse_value_split_all.keys():
        print(f'{key:<24}', f'{"kl_div":<6}',  " ".join(f"{x:.4f}" for x in kl_div_value_split_all[key]))
        print(f'{key+ "_var":<24}', f'{"kl_div":<6}',  " ".join(f"{x:.4f}" for x in kl_div_value_split_all_vars[key]))
    for key in mse_value_split_all.keys():
        print(f'{key:<24}', f'{"pearsonr":<6}', "".join(f"{pearsonr_value_split_all[key]:.4f}"))
        print(f'{key+ "_var":<24}', f'{"pearsonr":<6}', "".join(f"{pearsonr_value_split_all_vars[key]:.4f}"))
    for key in mse_value_split_all.keys():
        print(f'{key:<24}', f'{"spearman":<6}', "".join(f"{spearman_value_split_all[key]:.4f}"))
        print(f'{key+ "_var":<24}', f'{"spearman":<6}', "".join(f"{spearman_value_split_all_vars[key]:.4f}"))

    print('end_print\n\n')


def load_pretrain_weights_local(DEVICE, TRAIN_FROM_SCRATCH=True, RETRAIN=False, dataname='cifar10', 
                          train_loader=None, test_loader=None, 
                          checkpoints_folder="LIRA_checkpoints", SUFFIX=None, resume=False, SEED=42,
                          BREAKs=None, breakcount=0, shadow_folder='LIRA_checkpoints/shadow_models/', shadow_idx=0):
    """
    directly download weights of a model trained exclusively on the retain set
    """
    if RETRAIN:
        local_path = f"retrain_weights_resnet18_{dataname}_{SUFFIX}_seed_{SEED}.pth" if SUFFIX else f"retrain_weights_resnet18_{dataname}_seed_{SEED}.pth"
    else:
        local_path = f"weights_resnet18_{dataname}.pth"

    if dataname == 'cifar10' or dataname == 'cinic10':
        num_classes = 10
    elif dataname == 'cifar100':
        num_classes = 100

    if dataname == 'cifar100' or dataname == 'cinic10':
        epochs = 201
    elif dataname == 'cifar10':
        epochs = 170

    break_epoch = breakcount * int(epochs/BREAKs) if BREAKs is not None else 0

    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)
    model_path = checkpoints_folder + '/unle_models/' + local_path if RETRAIN else checkpoints_folder + '/internal_models/' + local_path
    print(f"model_path: {model_path}")
    if (not os.path.exists(model_path) or (BREAKs is not None and not os.path.exists(model_path.split('.pth')[0] + f"_from{shadow_idx}_break_{break_epoch}.pth"))) and TRAIN_FROM_SCRATCH:
        # print the training time
        start_time = time.time()
        model = ResNet18(num_classes=num_classes)
        load_path = shadow_folder + f'{dataname}/shadow_model_{shadow_idx}.pth'
        shadow_weight = torch.load(
            load_path, map_location=torch.device(DEVICE))
        model.load_state_dict(shadow_weight)
        # model = resnet18(weights=None, num_classes=num_classes).to(DEVICE)
        model.to(DEVICE)

        model.eval()
        init_acc = accuracy(model, train_loader, DEVICE)
        print(
            f" | init accuracy: {init_acc} |")
    
        if resume:
            print('==> Resuming from checkpoint..')
            assert os.path.isdir(
                checkpoints_folder), 'Error: no checkpoints directory found!'
            checkpoint = torch.load(checkpoints_folder + '/ckpt.pth')
            model.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']

        optimizer = optim.SGD(model.parameters(), lr=0.1,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                # print("loss: ", loss)

                optimizer.step()
            scheduler.step()

            model.eval()
            acc = accuracy(model, test_loader, DEVICE)
            train_acc = accuracy(model, train_loader, DEVICE)
            print(
                f"Epoch {epoch+1} | Validation accuracy: {acc} Train accuracy: {train_acc} | Train time: {time.time() - start_time}")

            if acc > best_acc:
                # print('Saving..')
                state = {
                    'net': model.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                # torch.save(state, checkpoints_folder + '/ckpt.pth')
                best_acc = acc

            if (BREAKs is not None) and (epoch % int(epochs/BREAKs) == 0 ):
                # save model weights 
                torch.save(state['net'], model_path.split('.pth')[0] + f"_from{shadow_idx}_break_{epoch}.pth")

        # save model weights
        model.to('cpu')
        # state['net'].to('cpu')
        torch.save(state['net'], model_path)
        # clean up
        del model, optimizer, scheduler, criterion

    if BREAKs is not None:
        weights_pretrained = torch.load(
            model_path.split('.pth')[0] + f"_from{shadow_idx}_break_{break_epoch}.pth", map_location=torch.device(DEVICE))
        return weights_pretrained, model_path
    else:
        weights_pretrained = torch.load(
            model_path, map_location=torch.device(DEVICE))

        return weights_pretrained, model_path

def train_classifier_local(train_loader, dataname, test_loader, TYPE, DEVICE, 
                     checkpoints_folder='LIRA_checkpoints', SUFFIX=None, 
                     PRIVACY=False, SEED=42, BREAKs=None, breakcount=0, shadow_folder='LIRA_checkpoints/shadow_models/', shadow_idx=0):
    
    if TYPE == 'ori':
        model_path = checkpoints_folder + '/internal_models/'+ 'weights_{}.pth'.format(dataname) if not PRIVACY else checkpoints_folder + '/weights_{}_privacy.pth'.format(dataname)
            
    elif TYPE == 'retrain':
        model_path = checkpoints_folder + f'/unle_models/retrain_weights_{dataname}_{SUFFIX}_seed_{SEED}.pth' 
    
    if dataname == 'location':
        model = LocationClassifier()
        lr = 0.01
        epochs = 81
    elif dataname == 'texas':
        model = TexasClassifier()
        lr = 0.01
        epochs = 51
    elif dataname == 'purchase':
        model = PurchaseClassifier()
        lr = 0.01
        epochs = 71

    # epoch 80 for purchase
    # epoch 60 for texas
    
    
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    break_epoch = breakcount * int(epochs/BREAKs) if BREAKs is not None else 0

    print(f"model_path: {model_path}")
    if not os.path.exists(model_path) or (BREAKs is not None and not os.path.exists(model_path.split('.pth')[0] + f"_from{shadow_idx}_break_{break_epoch}.pth")):
        criterion = nn.CrossEntropyLoss()
        load_path = shadow_folder + f'{dataname}/shadow_model_{shadow_idx}.pth'
        shadow_weight = torch.load(
            load_path, map_location=torch.device(DEVICE))
        model.load_state_dict(shadow_weight)
        model = model.to(DEVICE)
        model.eval()
        init_acc = accuracy(model, train_loader, DEVICE)
        print(
            f" | init accuracy: {init_acc} |")

        if not PRIVACY:
            start_time = time.time()
            optimizer = optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

            best_acc = 0
                
            for epoch in range(epochs):
                model.train()

                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()

                    optimizer.step()
                scheduler.step()

                model.eval()
                acc = accuracy(model, test_loader, DEVICE)
                train_acc = accuracy(model, train_loader, DEVICE)
                print(
                    f"Epoch {epoch+1} | Validation accuracy: {acc} Train accuracy: {train_acc} | Train time: {time.time() - start_time}")

                if acc > best_acc:
                    print('Saving..')
                    state = {
                        'net': model.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                    }
                    best_acc = acc

                if (BREAKs is not None) and (epoch % int(epochs/BREAKs) == 0 ):
                    # save model weights 
                    torch.save(state['net'], model_path.split('.pth')[0] + f"_from{shadow_idx}_break_{epoch}.pth")
                    print(f"internal model saved as: {model_path.split('.pth')[0] + f'_from{shadow_idx}_break_{epoch}.pth'}")

            # save model weights
            model.to('cpu')
            # state['net'].to('cpu')
            torch.save(state['net'], model_path)
            model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
            # clean up
            del optimizer, scheduler, criterion
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    
    if BREAKs is not None:
        model.load_state_dict(torch.load(model_path.split('.pth')[0] + f"_from{shadow_idx}_break_{break_epoch}.pth", map_location=torch.device(DEVICE)))
        return model
    else:
        return model
        
def parse_args():
    parser = argparse.ArgumentParser(description="Test UnleScore on various datasets and unlearn methods.")
    
    parser.add_argument('--dataname', type=str, default=None, choices=['cifar10', 'cifar100', 'cinic10', 'purchase'],
                        help="Dataset to use")
    parser.add_argument('--unlearn_method', type=str, default='retrain', 
                        choices=['retrain', 'finetune', 'ssd', 'fisher', 'forsaken', 'l_codec', 'ascent', 'boundary_expanding'],
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
    parser.add_argument('--init_idx', type=int, default=0)
    parser.add_argument('--INTERAPPROX', type=int, default=100)
    parser.add_argument('--BREAKs', type=int, default=20)
    parser.add_argument('--metrics', nargs='+', default=['p_Unleak', 'p_LiRA', 'p_update_LiRA', 'p_EMIA', 'p_EMIA_p', 'p_RMIA', 'p_SimpleDiff', 'p_LDiff', 'p_interapprox',  'p_random'],
                        help="Metrics to evaluate")
    parser.add_argument('--MODEL_CHECK', action='store_true', 
                        help="Whether to use exact flag during unlearning")

    
    return parser.parse_args()

def state_dict_to_cpu(sdict, device = "cpu"):
    for k, v in sdict.items():
        sdict[k] = sdict[k].to(device)
    return sdict


def list_to_cpu(slist, device = "cpu"):
    for i in range(len(slist)):
        slist[i] = slist[i].to(device)
    return slist

# main function
if __name__ == "__main__":
    args_input = parse_args()

    if None == args_input.dataname:
        for dataname in ['location','cifar10', 'cinic10',  'cifar100', 'purchase', 'texas' ]:
        # for dataname in ['cifar100']:
            args_input.dataname = dataname
            get_scores(args_input)
    else:
        get_scores(args_input)
        # args.update({'unlearn_type': 'class_percentage', 'forget_class_ratio': 0.5})
        # audit_scores, audit_scores_rt, nonmem_records, forget_records, nonmem_records_rt, retain_records = get_scores(args)
        # args.update({'unlearn_type': 'one_class'})
        # audit_scores, audit_scores_rt, nonmem_records, forget_records, nonmem_records_rt, retain_records = get_scores(args)