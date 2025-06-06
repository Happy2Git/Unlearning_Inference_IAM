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
    
    # Check if all logits are saved, if not, save them
    records_folder = args.records_folder + f'seed_{SEED}/'
    records_path = load_records(records_folder, args, batch_size=batch_size, DEVICE=DEVICE, SEED=SEED)
    # return

    # for metrics in args.metrics:
    #     if metrics not in ['p_Unleak', 'p_LiRA', 'p_update_LiRA', 'p_EMIA', 'p_RMIA', 'p_SimpleDiff', 'p_Dratio']:
    #         raise ValueError(f"Metrics {metrics} is not supported")

    audit_predicts = {}

    if 'p_LiRA' in args.metrics:
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
        fix_variance = True if len(args.SHADOW_IDXs) <= 1 else False
        if fix_variance:
            std_shadow_fix = shadow_target_ori_carlogis.std()
            std_shadow_target_out_carlogis = np.full_like(std_shadow_target_out_carlogis, std_shadow_fix)

        if 'p_LiRA' in args.metrics:
            p_LiRA = 1-norm.cdf(target_unl_carlogi, loc=mean_shadow_target_out_carlogis,
                                    scale=std_shadow_target_out_carlogis+1e-30)
            audit_predicts['p_LiRA'] = p_LiRA

            mean_shadow_target_in_carlogis = target_ori_carlogi.numpy()
            std_shadow_target_in_carlogis = torch.zeros_like(target_ori_carlogi).numpy()
            if True:
                std_shadow_fix = target_ori_carlogi.numpy().std()
                std_shadow_target_in_carlogis = np.full_like(std_shadow_target_in_carlogis, std_shadow_fix)

            p_in = norm.logpdf(target_unl_carlogi, loc=mean_shadow_target_in_carlogis,
                                scale=std_shadow_target_in_carlogis+1e-30)
            p_out = norm.logpdf(target_unl_carlogi, loc=mean_shadow_target_out_carlogis,
                                    scale=std_shadow_target_out_carlogis+1e-30)
            p_LiRA_Online = (- (p_in - p_out))

            audit_predicts['p_LiRA_Online'] = p_LiRA_Online
            print(f"p_LiRA done")

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

    if 'p_RMIA' in args.metrics:
        target_unl_logits = torch.load(records_path['target_unl_logits'])
        target_ori_logits = torch.load(records_path['target_ori_logits'])
        target_labels = torch.load(records_path['target_labels'])
        ref_unl_logits = torch.load(records_path['ref_unl_logits'])
        ref_labels = torch.load(records_path['ref_labels'])
        target_shadow_logits = torch.load(records_path['target_shadow_logits'])
        ref_shadow_logits = torch.load(records_path['ref_shadow_logits'])


        ref_in_logit_pop = torch.load(records_path['ref_popin_logits'])

        p_RMIA = 1-rmia(target_unl_logits, target_shadow_logits, target_labels, 
                ref_unl_logits, ref_shadow_logits, ref_labels, 
                OFFLINE=False, ref_in_logit_target=target_ori_logits, ref_in_logit_pop=ref_in_logit_pop,
                model_list=args.SHADOW_IDXs, metric='taylor-soft-margin', dataname= args.dataname, DEVICE=DEVICE)

        audit_predicts['p_RMIA_online'] = p_RMIA

        # p_RMIA = 1-rmia_old(target_set, unl_model, ref_set, 
        #         shadow_path, ori_model, ref_in_model, 
        #         OFFLINE=False,
        #         model_numbers=args.model_numbs, metric='taylor-soft-margin', dataname= args.dataname, DEVICE=DEVICE)

        # audit_predicts['p_RMIA_old_online'] = p_RMIA

        print(f"p_RMIA done")

    target_ori_logits = torch.load(records_path['target_ori_logits'])
    target_unl_logits = torch.load(records_path['target_unl_logits'])
    target_labels = torch.load(records_path['target_labels'])

    target_unl_proby = model_confidence(target_unl_logits, target_labels)

    target_unlearn_flags = torch.load(records_path['target_unlearn_flags'])


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

        loss = torch.nn.CrossEntropyLoss(reduction="none", reduce=False)
        target_unl_lossy = loss(target_unl_logits, target_labels)
        target_ori_lossy = loss(target_ori_logits, target_labels)
        shadow_target_lossy = []
        for ref_idx in args.SHADOW_IDXs:
            shadow_signal_target = loss(target_ref_logits[:,ref_idx,:].squeeze(), target_labels)
            shadow_target_lossy.append((shadow_signal_target))
        shadow_target_lossy = torch.stack(shadow_target_lossy)

        eps1 = 1e-2 if args.dataname != 'texas' else 1e-1
        eps2 = 1e-5
        max_value =  -np.log(eps1-np.log(eps2+1))
        min_value = -np.log(eps1-np.log(eps2+0))
        target_unl_logloss = (-np.log(eps1-np.log(eps2+target_unl_proby))-min_value)/(max_value-min_value)
        target_ori_logloss = (-np.log(eps1-np.log(eps2+target_ori_proby))-min_value)/(max_value-min_value)
        shadow_target_logloss = (-np.log(eps1-np.log(eps2+shadow_target_proby))-min_value)/(max_value-min_value)

        for INTER_APPRO in [100]:
            score_ori_weighted, score_unl_weighted = interpolate_appro(target_ori_logloss, target_unl_logloss, shadow_target_logloss, INTER_APPRO)
            key = f'p_IPapprox_{INTER_APPRO}_ol'
            audit_predicts[key] = score_unl_weighted

            # score_ori_weighted, score_unl_weighted = interpolate_appro(target_ori_logloss_wonorm, target_unl_logloss_wonorm, shadow_target_logloss_wonorm, INTER_APPRO)
            # key = f'p_IPapprox_{INTER_APPRO}_ol_wonorm'
            # audit_predicts[key] = score_unl_weighted

            # score_ori_weighted, score_unl_weighted = interpolate_appro(target_ori_logloss_new, target_unl_logloss_new, shadow_target_logloss_new, INTER_APPRO)
            # key = f'p_IPapprox_{INTER_APPRO}_ol_new'
            # audit_predicts[key] = score_unl_weighted
            print(f"{key} done")
        eps1= 1e-1 if args.dataname == 'texas' else 1e-2

        for eps1 in [10,1,1e-1,1e-2]:
            for eps2 in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9,1e-10]:
                if np.exp(eps1)-1<eps2:
                    continue
                target_unl_logloss_wonorm_new = (-np.log(eps1-np.log(eps2+target_unl_proby)))
                target_ori_logloss_wonorm_new = (-np.log(eps1-np.log(eps2+target_ori_proby)))
                shadow_target_logloss_wonorm_new = (-np.log(eps1-np.log(eps2+shadow_target_proby)))
                score_ori_weighted, score_unl_weighted = interpolate_appro(target_ori_logloss_wonorm_new, target_unl_logloss_wonorm_new, shadow_target_logloss_wonorm_new, INTER_APPRO)
                key = f'p_IPapprox_{eps1}_{eps2}_ol_loss'
                audit_predicts[key] = score_unl_weighted
                print(f"{key} done")
        # INTER_APPRO = 100
        # score_ori_weighted, score_unl_weighted = interpolate_appro(target_ori_logloss, target_unl_logloss, shadow_target_logloss, INTER_APPRO)
        # key = f'p_IPapprox_{INTER_APPRO}_ol'
        # audit_predicts[key] = score_unl_weighted

        # INTER_APPRO = 2
        # score_ori_weighted, score_unl_weighted = interpolate_appro_test(target_ori_logloss_wonorm_new, target_unl_logloss_wonorm_new, shadow_target_logloss_wonorm_new, INTER_APPRO, type = 'gumbel')
        # key = f'p_IPapprox_{INTER_APPRO}_ol_test'
        # audit_predicts[key] = score_unl_weighted
        # key = f'p_IPapprox_{INTER_APPRO}_ol_test_diff'
        # audit_predicts[key] = score_unl_weighted-score_ori_weighted

        print(f"{key} done")

    return audit_predicts, target_unlearn_flags, target_unl_proby


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
    if not os.path.exists(args.records_folder + 'reseed/'):
        os.makedirs(args.records_folder + 'reseed/')

    dataname = args.dataname
    unlearn_method = args.unlearn_method
    unlearn_type = args.unlearn_type
    
    SUFFIX = args.SUFFIX
    print(SUFFIX)
    target_ori_logits_path = args.records_folder + f'reseed/{dataname}_target_ori_logits_seed_{SEED}.pth' # #number x #class
    target_unl_logits_path = records_folder + f'{SUFFIX}_target_unl_logits.pth' # #number x #class
    target_labels_path = args.records_folder + f'reseed/{dataname}_target_labels_seed_{SEED}.pth'  # #number
    target_unlearn_flags_path = records_folder + f'{SUFFIX}_target_unlearn_flags.pth'  # #number
    target_shadow_logits_path = args.records_folder + f'reseed/{dataname}_target_shadows_logits_seed_{SEED}.pth' # #number x #shadow x #class
    ref_ori_logits_path = records_folder + f'{dataname}_ref_ori_logits.pth' # #number x #class
    ref_unl_logits_path = records_folder + f'{SUFFIX}_ref_unl_logits.pth' # #number x #class
    ref_labels_path = records_folder + f'{dataname}_ref_labels.pth' # #number x #class
    ref_shadow_logits_path = records_folder + f'{dataname}_ref_shadow_logits.pth' # #number x #ref x #class
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
        return records_path
    
    ori_model, unl_model, target_set, unlearn_flags, ref_set, test_loader, shadow_path, shadow_set, SUFFIX, accuracy = fetch_data_model(args, verbose=True, SEED=SEED)

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
    
    if not os.path.exists(target_shadow_logits_path) or not os.path.exists(ref_shadow_logits_path):
        shadow_model_list = []
        dataname = args.dataname
        if dataname == 'cifar10' or dataname == 'cinic10':
            num_classes = 10
        elif dataname == 'location':
            num_classes = 30
        elif dataname == 'cifar100' or dataname == 'texas' or dataname == 'purchase':
            num_classes = 100

        for shadow_idx in args.SHADOW_IDXs:
            load_path = args.shadow_folder + f'{dataname}/shadow_model_{shadow_idx}.pth'
            shadow_weight = torch.load(
                load_path, map_location=torch.device(DEVICE))
            if dataname == 'cifar10' or dataname == 'cifar100' or dataname == 'cinic10':
                shadow_model = ResNet18(num_classes=num_classes)
            elif dataname == 'purchase':
                shadow_model = PurchaseClassifier()
            shadow_model.load_state_dict(shadow_weight)
            shadow_model.to(DEVICE)
            shadow_model.eval()
            shadow_model_list.append(shadow_model)

        if not os.path.exists(target_shadow_logits_path):
            target_shadow_logits = []
            target_loader = DataLoader(target_set, batch_size=batch_size, shuffle=False, num_workers=4)
            with torch.no_grad():
                for x, y in target_loader:
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)
                    shadow_logit = torch.stack([shadow_model(x).cpu() for shadow_model in shadow_model_list], dim=1)
                    target_shadow_logits.append(shadow_logit)

            target_shadow_logits = torch.cat(target_shadow_logits)
            print(f'shadow_logit shape: {shadow_logit.shape} target_shadow_logits shape: {target_shadow_logits.shape}')

            torch.save(target_shadow_logits, target_shadow_logits_path)
            print(f"Save target_shadow_logits_path to {records_folder}")
        
        if not os.path.exists(ref_shadow_logits_path):
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


def fetch_data_model(args,  verbose=False, SEED = 42):
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
    shadow_training(shadow_set, dataname, ratio=0.8, ORIGIN=True, shadow_list=args.SHADOW_IDXs, shadow_path=shadow_path, DEVICE=DEVICE, VERBOSE=verbose)

    if dataname == 'cifar10' or dataname == 'cifar100'  or dataname == 'cinic10':
        weights_pretrained, _ = load_pretrain_weights(DEVICE, TRAIN_FROM_SCRATCH=True, RETRAIN=True, dataname=dataname,
                                                train_loader=init_train_loader, test_loader=val_loader, checkpoints_folder='LIRA_checkpoints', SEED=SEED)
        # load model with pre-trained weights
        model = ResNet18(num_classes=num_classes)
        model.load_state_dict(weights_pretrained)
        model.to(DEVICE)
        model.eval()
    else:
        model = train_classifier(init_train_loader, dataname, val_loader, 'ori', DEVICE, checkpoints_folder='LIRA_checkpoints', SEED=SEED, ReSEED=True)
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
            ul_model = train_classifier(retain_loader, dataname, 'unlearn', DEVICE, checkpoints_folder='LIRA_checkpoints', SUFFIX=SUFFIX, SEED=SEED)
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


def get_scores(args_input):
    args = copy.deepcopy(args_input)
    # args.SHADOW_IDXs = random.sample(range(128), args.model_numbs)
    args.SHADOW_IDXs = list(range(args.model_numbs))
    if args.unlearn_type == 'set_random':
        args.SUFFIX = args.dataname + '_' + args.unlearn_method + '_' + args.unlearn_type + '_' + str(args.forget_size)
        SEED_init = args.SEED_init
        RNG = torch.Generator()
        RNG.manual_seed(SEED_init)
        Scores = {}
        Unlearn_flags = []
        Unlearn_proby = []
        cross_model_stat0 = {}
        cross_model_stat1 = {}
        for loop in range(args.LOOP):
            SEED = SEED_init + loop       
            print(f"Loop {loop}, SEED {SEED}")         
            # audit_predicts, target_unlearn_flags, target_unl_proby = audit_nonmem_mono(args, DEVICE=DEVICE, SEED=SEED)
            records_folder = args.records_folder + f'seed_{SEED}/'
            records_path = load_records(records_folder, args, batch_size=1024, DEVICE=DEVICE, SEED=SEED)
            print(records_path)

        print('end_print\n\n')

    else:
        SEED_init = args.SEED_init
        RNG = torch.Generator()
        RNG.manual_seed(SEED_init)
        Scores = {}
        Unlearn_flags = []
        Unlearn_proby = []
        cross_model_stat0 = {}
        cross_model_stat1 = {}
        for loop in range(args.LOOP):
            args.forget_class = args.CLASS_init+loop  
            print(f"Loop {loop}, SEED {SEED_init}, CLASS {args.forget_class}, TYPE {args.unlearn_type}")         
            if args.unlearn_type == 'one_class':
                args.SUFFIX = args.dataname + '_' + args.unlearn_method + '_' + args.unlearn_type + '_' + str(args.forget_class)
            elif args.unlearn_type == 'class_percentage':
                args.SUFFIX = args.dataname + '_' + args.unlearn_method + '_' + args.unlearn_type + '_' + 'class' + str(args.forget_class) + '_' + str(args.forget_class_ratio)
            
            records_folder = args.records_folder + f'seed_{SEED}/'
            records_path = load_records(records_folder, args, batch_size=1024, DEVICE=DEVICE, SEED=SEED)
            print(records_path)

        print('end_print\n\n')

    
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
    parser.add_argument('--model_numbs', type=int, default=128, 
                        help="Number of models to train")
    parser.add_argument('--LOOP', type=int, default=5, 
                        help="Number of training loops")
    parser.add_argument('--EXACT_FLAG', action='store_true', 
                        help="Whether to use exact flag during unlearning")
    parser.add_argument('--shadow_folder',type=str, default='LIRA_checkpoints/shadow_models/')
    parser.add_argument('--records_folder',type=str, default='LIRA_checkpoints/records/')
    parser.add_argument('--SEED_init', type=int, default=42)
    parser.add_argument('--CLASS_init', type=int, default=0)
    parser.add_argument('--INTERAPPROX', type=int, default=100)
    # parser.add_argument('--metrics', nargs='+', default=[ 'p_LiRA', 'p_EMIA_p','p_Unleak','p_update_LiRA', 'p_EMIA', 'p_RMIA', 'p_SimpleDiff', 'p_LDiff', 'p_interapprox', 'p_interapprox_init'],
    #                     help="Metrics to evaluate")
    parser.add_argument('--metrics', nargs='+', default=[ 'p_LiRA'],
                        help="Metrics to evaluate")
    parser.add_argument('--MODEL_CHECK', action='store_true', 
                        help="Whether to use exact flag during unlearning")

    
    return parser.parse_args()

# main function
if __name__ == "__main__":
    args_input = parse_args()

    if None == args_input.dataname:
        for dataname in ['cifar10', 'cinic10',  'cifar100', 'purchase']:
        # for dataname in ['cifar100']:
            args_input.dataname = dataname
            get_scores(args_input)
    else:
        get_scores(args_input)
        # args.update({'unlearn_type': 'class_percentage', 'forget_class_ratio': 0.5})
        # audit_scores, audit_scores_rt, nonmem_records, forget_records, nonmem_records_rt, retain_records = get_scores(args)
        # args.update({'unlearn_type': 'one_class'})
        # audit_scores, audit_scores_rt, nonmem_records, forget_records, nonmem_records_rt, retain_records = get_scores(args)