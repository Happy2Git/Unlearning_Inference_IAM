from unlearning_lib.models.resnet import ResNet18, ResNet34, ResNet50
# 'vgg16', 'vgg11', 'shufflenet', 'DenseNet', 'MobileNetV2', 'EfficientNet', 'swin_t']
from torchvision.models import vgg11, vgg16, shufflenet_v2_x1_0, densenet121, mobilenet_v2, efficientnet_b0, swin_t
from unlearning_lib.utils import accuracy
from unlearning_lib.models.purchase_classifier import PurchaseClassifier
from torchvision import transforms

from argparse import ArgumentParser
import os
import torch
# import list and warnings
from typing import List
import time
from torch import nn, optim
from data_loader import get_data_loaders, get_data_loaders_continual

import numpy as np

import random
from torch.nn.utils import parameters_to_vector as p2v
import copy
from torch.utils.data.dataset import Subset

def shadow_training(shadow_set, dataname='cifar10', ratio=0.8, shadow_nums = 2, shadow_list = None, 
                    ORIGIN=True, batch_size=128, num_workers=2, shuffle=True, SEED=42, 
                    shadow_path = 'LIRA_checkpoints/shadow_models/cifar10', DEVICE='cuda', VERBOSE=False,
                    ARCH=False, arch_name='ResNet18'):
    # shadow training
    RNG = torch.Generator()
    RNG.manual_seed(SEED)
    if not os.path.exists(shadow_path):
        os.makedirs(shadow_path)
    if dataname == 'cifar10' or dataname == 'cinic10':
        lr = 0.1
    elif dataname == 'cifar100':
        lr = 0.1
    elif dataname == 'location':
        lr = 0.01
    elif dataname == 'texas':
        lr = 0.01
    elif dataname == 'purchase':
        lr = 0.01
    

    if shadow_list is None:
        shadow_list = range(shadow_nums)

    print(f'{dataname} shadow_origin training')
    # load shadow model
    load_path = shadow_path + f'/shadow_origin_model.pth'
    load_path_logit = shadow_path + f'/shadow_origin_model_logit.pth'
    load_path_label = shadow_path + f'/shadow_origin_model_label.pth'

    if ARCH:
        load_path = load_path.split('.pth')[0] + f"arch_{arch_name}.pth"
        load_path_logit = load_path_logit.split('.pth')[0] + f"arch_{arch_name}.pth"
        load_path_label = load_path_label.split('.pth')[0] + f"arch_{arch_name}.pth"

    os.makedirs(os.path.dirname(shadow_path), exist_ok=True)
    if ORIGIN and not os.path.exists(load_path):
        shadow_loader = torch.utils.data.DataLoader(shadow_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=RNG)

        start_time = time.time()
        if dataname == 'cifar10' or dataname == 'cinic10':
            shadow_model = ResNet18(num_classes=10)
        elif dataname == 'cifar100':
            shadow_model = ResNet18(num_classes=100)
        elif dataname == 'purchase':
            shadow_model = PurchaseClassifier()

        epochs_set=-1
        if ARCH and dataname == 'cifar100':
            if arch_name == 'ResNet18':
                shadow_model = ResNet18(num_classes=100)
            elif arch_name == 'ResNet34':
                shadow_model = ResNet34(num_classes=100)
            elif arch_name == 'ResNet50':
                shadow_model = ResNet50(num_classes=100)
                epochs_set = 150
            elif arch_name == 'vgg16':
                shadow_model = vgg16(weights=None)
                shadow_model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=100)
                lr = 0.01
            elif arch_name == 'vgg11':
                shadow_model = vgg11(weights=None)
                shadow_model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=100)
                lr = 0.01
            elif arch_name == 'shufflenet': # RuntimeError: Calculated padded input size per channel: (1 x 1). Kernel size: (3 x 3). Kernel size can't be greater than actual input size
                shadow_model = shufflenet_v2_x1_0(weights=None)
                epochs_set = 200
                # Modify classifier for CIFAR100
                shadow_model.fc = torch.nn.Linear(shadow_model.fc.in_features, 100)
            elif arch_name == 'DenseNet':
                shadow_model = densenet121(weights=None)
                shadow_model.classifier = torch.nn.Linear(in_features=1024, out_features=100)
            elif arch_name == 'MobileNetV2':
                shadow_model = mobilenet_v2(weights=None)
                shadow_model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=100)
                epochs_set = 300
            elif arch_name == 'EfficientNet':
                shadow_model = efficientnet_b0(weights=None)
                shadow_model._fc = torch.nn.Linear(in_features=1280, out_features=100)
                epochs_set = 300
            elif arch_name == 'swin_t':
                shadow_model = swin_t(weights=None)  # Initialize without pretrained weights
                # Modify head for CIFAR100 (100 classes)
                shadow_model.head = torch.nn.Linear(shadow_model.head.in_features, 100)
                epochs_set = 300


        shadow_model.to(DEVICE)
        shadow_model.train()
        # define optimizer

        optimizer = optim.SGD(shadow_model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4)
            
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0
        epochs = 201 if epochs_set == -1 else epochs_set
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}')
            shadow_model.train()
            for inputs, targets in shadow_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = shadow_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                # print("loss: ", loss)
                optimizer.step()
            scheduler.step()
            shadow_model.eval()
            acc = accuracy(shadow_model, shadow_loader, DEVICE)
            print(
                f"Epoch {epoch+1} | Training accuracy: {acc} | Train time: {time.time() - start_time}")

            if acc > best_acc:
                if VERBOSE:
                    print('Saving..')

                state = {
                    'net': shadow_model.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                best_acc = acc

        shadow_model.to('cpu')
        # state['net'].to('cpu')
        # save shadow model
        torch.save(state['net'], load_path)
        del optimizer, scheduler, criterion
        print(f'{dataname} shadow_origin training finished & saved')

    if ORIGIN and (not os.path.exists(load_path_logit) or not os.path.exists(load_path_label)):
        if dataname == 'cifar10' or dataname == 'cinic10':
            shadow_model = ResNet18(num_classes=10)
        elif dataname == 'cifar100':
            shadow_model = ResNet18(num_classes=100)
        elif dataname == 'purchase':
            shadow_model = PurchaseClassifier()
        if ARCH and dataname == 'cifar100':
            if arch_name == 'ResNet18':
                shadow_model = ResNet18(num_classes=100)
            elif arch_name == 'ResNet34':
                shadow_model = ResNet34(num_classes=100)
            elif arch_name == 'ResNet50':
                shadow_model = ResNet50(num_classes=100)
            elif arch_name == 'vgg16':
                shadow_model = vgg16(weights=None)
                shadow_model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=100)
                lr = 0.01
            elif arch_name == 'vgg11':
                shadow_model = vgg11(weights=None)
                shadow_model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=100)
                lr = 0.01
            elif arch_name == 'shufflenet': # RuntimeError: Calculated padded input size per channel: (1 x 1). Kernel size: (3 x 3). Kernel size can't be greater than actual input size
                shadow_model = shufflenet_v2_x1_0(weights=None)
                # Modify classifier for CIFAR100
                shadow_model.fc = torch.nn.Linear(shadow_model.fc.in_features, 100)
            elif arch_name == 'DenseNet':
                shadow_model = densenet121(weights=None)
                shadow_model.classifier = torch.nn.Linear(in_features=1024, out_features=100)
            elif arch_name == 'MobileNetV2':
                shadow_model = mobilenet_v2(weights=None)
                shadow_model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=100)
            elif arch_name == 'EfficientNet':
                shadow_model = efficientnet_b0(weights=None)
                shadow_model._fc = torch.nn.Linear(in_features=1280, out_features=100)
            elif arch_name == 'swin_t':
                shadow_model = swin_t(weights=None)  # Initialize without pretrained weights
                # Modify head for CIFAR100 (100 classes)
                shadow_model.head = torch.nn.Linear(shadow_model.head.in_features, 100)

        shadow_model.load_state_dict(torch.load(load_path))
        shadow_model.to(DEVICE)
        shadow_model.eval()
        shadow_loader = torch.utils.data.DataLoader(shadow_set, batch_size=batch_size*2, shuffle=False, num_workers=num_workers)
        logit_list = []
        label_list = []
        with torch.no_grad():
            for inputs, targets in shadow_loader:
                inputs = inputs.to(DEVICE)
                logit_list.append(shadow_model(inputs).cpu())
                label_list.append(targets.cpu())

        logit_list = torch.cat(logit_list)
        label_list = torch.cat(label_list)
        torch.save(logit_list, load_path_logit)
        torch.save(label_list, load_path_label)
        print(f'{dataname} shadow_origin logit & prob saved')
        shadow_model.to('cpu')
        del shadow_model

    RNG = torch.Generator()
    RNG.manual_seed(SEED)
    for shadow_idx in shadow_list:
        # load shadow model
        load_path = shadow_path + f'/shadow_model_{shadow_idx}.pth'
        load_path_logit_train = shadow_path + f'/shadow_model_{shadow_idx}_logit_train.pth'
        load_path_label_train = shadow_path + f'/shadow_model_{shadow_idx}_label_train.pth'
        load_path_logit_test = shadow_path + f'/shadow_model_{shadow_idx}_logit_test.pth'
        load_path_label_test = shadow_path + f'/shadow_model_{shadow_idx}_label_test.pth'

        if ARCH:
            load_path = load_path.split('.pth')[0] + f"arch_{arch_name}.pth"
            load_path_logit_train = load_path_logit_train.split('.pth')[0] + f"arch_{arch_name}.pth"
            load_path_label_train = load_path_label_train.split('.pth')[0] + f"arch_{arch_name}.pth"
            load_path_logit_test = load_path_logit_test.split('.pth')[0] + f"arch_{arch_name}.pth"
            load_path_label_test = load_path_label_test.split('.pth')[0] + f"arch_{arch_name}.pth"

        if not os.path.exists(load_path) or not os.path.exists(load_path_logit_train) or not os.path.exists(load_path_logit_test):
            # split shadow set
            FIX_TRAIN_SIZE = 200
            TRAIN_SHADOW_SIZE = int(len(shadow_set)*ratio)-FIX_TRAIN_SIZE
            # SPLIT_LIST = [TRAIN_SHADOW_SIZE, len(shadow_set)-TRAIN_SHADOW_SIZE]
            RNG.manual_seed(RNG.get_state()[0].item() + shadow_idx)

            shadow_indices = shadow_set.indices  # get the indices of the new train set in the original dataset
            # print(f'len of shadow_set: {len(shadow_indices)}, {shadow_indices[-10:]}')
            shuffled_idx = torch.randperm(len(shadow_indices)-FIX_TRAIN_SIZE, generator=RNG)
            shadow_train_idx = shuffled_idx[:TRAIN_SHADOW_SIZE]
            # merge with fixed train set
            shadow_train_idx = torch.cat((shadow_train_idx, torch.tensor(range((len(shadow_indices)-FIX_TRAIN_SIZE), len(shadow_indices)))), 0)
            shadow_test_idx = shuffled_idx[(TRAIN_SHADOW_SIZE):]
            shadow_train_set = torch.utils.data.Subset(shadow_set, shadow_train_idx)
            shadow_test_set = torch.utils.data.Subset(shadow_set, shadow_test_idx)

        if not os.path.exists(load_path):
            print(f'{dataname} shadow_idx:{shadow_idx} training')
            # shadow_train_set, shadow_test_set = torch.utils.data.random_split(shadow_set, SPLIT_LIST, generator=RNG)
            shadow_train_loader = torch.utils.data.DataLoader(shadow_train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=RNG)
            shadow_test_loader = torch.utils.data.DataLoader(shadow_test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=RNG)

            start_time = time.time()
            if dataname == 'cifar10' or dataname == 'cinic10':
                shadow_model = ResNet18(num_classes=10)
            elif dataname == 'cifar100':
                shadow_model = ResNet18(num_classes=100)
            elif dataname == 'purchase':
                shadow_model = PurchaseClassifier()

            epochs_set=-1
            if ARCH and dataname == 'cifar100':
                if arch_name == 'ResNet18':
                    shadow_model = ResNet18(num_classes=100)
                elif arch_name == 'ResNet34':
                    shadow_model = ResNet34(num_classes=100)
                elif arch_name == 'ResNet50':
                    shadow_model = ResNet50(num_classes=100)
                    epochs_set = 150
                elif arch_name == 'vgg16':
                    shadow_model = vgg16(weights=None)
                    shadow_model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=100)
                    lr = 0.01
                elif arch_name == 'vgg11':
                    shadow_model = vgg11(weights=None)
                    shadow_model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=100)
                    lr = 0.01
                elif arch_name == 'shufflenet': # RuntimeError: Calculated padded input size per channel: (1 x 1). Kernel size: (3 x 3). Kernel size can't be greater than actual input size
                    shadow_model = shufflenet_v2_x1_0(weights=None)
                    epochs_set = 200
                    # Modify classifier for CIFAR100
                    shadow_model.fc = torch.nn.Linear(shadow_model.fc.in_features, 100)
                elif arch_name == 'DenseNet':
                    shadow_model = densenet121(weights=None)
                    shadow_model.classifier = torch.nn.Linear(in_features=1024, out_features=100)
                elif arch_name == 'MobileNetV2':
                    shadow_model = mobilenet_v2(weights=None)
                    shadow_model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=100)
                    epochs_set = 300
                elif arch_name == 'EfficientNet':
                    shadow_model = efficientnet_b0(weights=None)
                    shadow_model._fc = torch.nn.Linear(in_features=1280, out_features=100)
                    epochs_set = 300
                elif arch_name == 'swin_t':
                    shadow_model = swin_t(weights=None)  # Initialize without pretrained weights
                    # Modify head for CIFAR100 (100 classes)
                    shadow_model.head = torch.nn.Linear(shadow_model.head.in_features, 100)
                    epochs_set = 300

            shadow_model.to(DEVICE)
            shadow_model.train()
            # define optimizer
            optimizer = optim.SGD(shadow_model.parameters(), lr=lr,
                                    momentum=0.9, weight_decay=5e-4)
            
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

            criterion = nn.CrossEntropyLoss()

            best_acc = 0
            epochs = 201 if epochs_set == -1 else epochs_set
            for epoch in range(epochs):
                shadow_model.train()
                for inputs, targets in shadow_train_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = shadow_model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    # print("loss: ", loss)
                    optimizer.step()
                scheduler.step()
                shadow_model.eval()
                acc = accuracy(shadow_model, shadow_test_loader, DEVICE)
                print(
                    f"Epoch {epoch+1} | Validation accuracy: {acc} | Train time: {time.time() - start_time}")

                if acc > best_acc:
                    if VERBOSE:
                        print('Saving..')

                    state = {
                        'net': shadow_model.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                    }
                    best_acc = acc

            shadow_model.to('cpu')
            # state['net'].to('cpu')
            # save shadow model
            torch.save(state['net'], load_path)
            del shadow_model, optimizer, scheduler, criterion
            print(f'{dataname} shadow_idx:{shadow_idx} training finished & saved')

        if not os.path.exists(load_path_logit_train) or not os.path.exists(load_path_logit_test):
            if dataname == 'cifar10' or dataname == 'cinic10':
                shadow_model = ResNet18(num_classes=10)
            elif dataname == 'cifar100':
                shadow_model = ResNet18(num_classes=100)
            elif dataname == 'purchase':
                shadow_model = PurchaseClassifier()

            if ARCH and dataname == 'cifar100':
                if arch_name == 'ResNet18':
                    shadow_model = ResNet18(num_classes=100)
                elif arch_name == 'ResNet34':
                    shadow_model = ResNet34(num_classes=100)
                elif arch_name == 'ResNet50':
                    shadow_model = ResNet50(num_classes=100)
                elif arch_name == 'vgg16':
                    shadow_model = vgg16(weights=None)
                    shadow_model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=100)
                elif arch_name == 'vgg11':
                    shadow_model = vgg11(weights=None)
                    shadow_model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=100)
                elif arch_name == 'shufflenet': # RuntimeError: Calculated padded input size per channel: (1 x 1). Kernel size: (3 x 3). Kernel size can't be greater than actual input size
                    shadow_model = shufflenet_v2_x1_0(weights=None)
                    # Modify classifier for CIFAR100
                    shadow_model.fc = torch.nn.Linear(shadow_model.fc.in_features, 100)
                elif arch_name == 'DenseNet':
                    shadow_model = densenet121(weights=None)
                    shadow_model.classifier = torch.nn.Linear(in_features=1024, out_features=100)
                elif arch_name == 'MobileNetV2':
                    shadow_model = mobilenet_v2(weights=None)
                    shadow_model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=100)
                elif arch_name == 'EfficientNet':
                    shadow_model = efficientnet_b0(weights=None)
                    shadow_model._fc = torch.nn.Linear(in_features=1280, out_features=100)
                elif arch_name == 'swin_t':
                    shadow_model = swin_t(weights=None)  # Initialize without pretrained weights
                    # Modify head for CIFAR100 (100 classes)
                    shadow_model.head = torch.nn.Linear(shadow_model.head.in_features, 100)

            shadow_model.load_state_dict(torch.load(load_path))
            shadow_model.to(DEVICE)
            shadow_model.eval()

            if not os.path.exists(load_path_logit_train):
                shadow_train_loader = torch.utils.data.DataLoader(shadow_train_set, batch_size=batch_size*2, shuffle=False, num_workers=num_workers)
                logit_list_train = []
                label_list_train = []
                with torch.no_grad():
                    for inputs, targets in shadow_train_loader:
                        inputs  = inputs.to(DEVICE) 
                        logit_list_train.append(shadow_model(inputs).cpu())
                        label_list_train.append(targets.cpu())

                logit_list_train = torch.cat(logit_list_train)
                label_list_train = torch.cat(label_list_train)
                torch.save(logit_list_train, load_path_logit_train)
                torch.save(label_list_train, load_path_label_train)
            
            if not os.path.exists(load_path_logit_test):
                logit_list_test = []
                label_list_test = []
                shadow_test_loader = torch.utils.data.DataLoader(shadow_test_set, batch_size=batch_size*2, shuffle=False, num_workers=num_workers)
                with torch.no_grad():
                    for inputs, targets in shadow_test_loader:
                        inputs = inputs.to(DEVICE) 
                        logit_list_test.append(shadow_model(inputs).cpu())
                        label_list_test.append(targets.cpu())

                logit_list_test = torch.cat(logit_list_test)
                label_list_test = torch.cat(label_list_test)
                torch.save(logit_list_test, load_path_logit_test)
                torch.save(label_list_test, load_path_label_test)

            print(f'{dataname} shadow_idx:{shadow_idx} logit & prob saved')
            shadow_model.to('cpu')
            del shadow_model

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('--dataname', type=str, default='cifar10')
    args.add_argument('--ratio', type=float, default=0.8)
    args.add_argument('--shadow_nums', type=int, default=128)
    args.add_argument('--shadow_idx', nargs='+', type=int, default=None)
    args.add_argument('--origin', type=bool, default=True)
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--num_workers', type=int, default=2)
    args.add_argument('--forget_size', type=int, default=500)
    args.add_argument('--shuffle', type=bool, default=True)
    args.add_argument('--RNG', type=int, default=42)
    args.add_argument('--shadow_path', type=str, default='LIRA_checkpoints/shadow_models/')
    args.add_argument('--DEVICE', type=str, default='cuda')
    args.add_argument('--VERBOSE', type=bool, default=False)
    args.add_argument('--ARCH', type=bool, default=False)
    args.add_argument('--arch_name', type=str, default='ResNet18')
    args = args.parse_args()
    print(args)

    RNG = torch.Generator()
    RNG.manual_seed(args.RNG)

    if args.ARCH and args.dataname == 'cifar100' and args.arch_name == 'shufflenet':
        def get_cifar_transforms(train=True):
            if train:
                transform = transforms.Compose([
                    transforms.Resize(224),  # Resize to ViT expected size
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(224, padding=4),
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5071, 0.4867, 0.4408), 
                        (0.2675, 0.2565, 0.2761)
                    )
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5071, 0.4867, 0.4408), 
                        (0.2675, 0.2565, 0.2761)
                    )
                ])
            return transform
        
        train_transforms = get_cifar_transforms(train=True)
        test_transforms = get_cifar_transforms(train=False)
 
    elif args.ARCH and args.dataname == 'cifar100' and args.arch_name == 'swin_t':
        def get_cifar_transforms(train=True):
            if train:
                transform = transforms.Compose([
                    transforms.Resize(224),  # Resize to ViT expected size
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(224, padding=4),
                    transforms.RandAugment(num_ops=2, magnitude=7),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5071, 0.4867, 0.4408), 
                        (0.2675, 0.2565, 0.2761)
                    ),
                    transforms.RandomErasing(p=0.25)
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5071, 0.4867, 0.4408), 
                        (0.2675, 0.2565, 0.2761)
                    )
                ])
            return transform
        
        train_transforms = get_cifar_transforms(train=True)
        test_transforms = get_cifar_transforms(train=False)
    else:
        train_transforms = None
        test_transforms = None

    _, _, _, _, _, shadow_set,_,_,_ = get_data_loaders(args.dataname, batch_size=args.batch_size, 
                                                       num_workers=args.num_workers, forget_size=args.forget_size, 
                                                       shuffle=args.shuffle,
                                                       train_transforms=train_transforms, test_transforms=test_transforms)
    # prepare shadowmodels
    shadow_path = args.shadow_path + args.dataname

    if args.shadow_idx is not None:
        shadow_list = args.shadow_idx
        print(f'shadow_list: {shadow_list}')
    else:
        shadow_list = range(args.shadow_nums)

    shadow_training(shadow_set, dataname=args.dataname, ratio=args.ratio, shadow_list=shadow_list, 
                    ORIGIN=args.origin, batch_size=args.batch_size, num_workers=args.num_workers, 
                    shuffle=args.shuffle, shadow_path = shadow_path, DEVICE=args.DEVICE, VERBOSE=args.VERBOSE,
                    ARCH=args.ARCH, arch_name=args.arch_name)
        
