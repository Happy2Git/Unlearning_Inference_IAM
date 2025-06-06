import torch
import numpy as np
import os
from torch import nn, optim
import random
import sys
import copy
import time
from .models.resnet import ResNet18
from unlearning_lib.models.purchase_classifier import PurchaseClassifier
def manual_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def display_progress(text, current_step, last_step, enabled=True,
                     fix_zero_start=True):
    """Draws a progress indicator on the screen with the text preceeding the
    progress

    Arguments:
        test: str, text displayed to describe the task being executed
        current_step: int, current step of the iteration
        last_step: int, last possible step of the iteration
        enabled: bool, if false this function will not execute. This is
            for running silently without stdout output.
        fix_zero_start: bool, if true adds 1 to each current step so that the
            display starts at 1 instead of 0, which it would for most loops
            otherwise.
    """
    if not enabled:
        return

    # Fix display for most loops which start with 0, otherwise looks weird
    if fix_zero_start:
        current_step = current_step + 1

    term_line_len = 80
    final_chars = [':', ';', ' ', '.', ',']
    if text[-1:] not in final_chars:
        text = text + ' '
    if len(text) < term_line_len:
        bar_len = term_line_len - (len(text)
                                   + len(str(current_step))
                                   + len(str(last_step))
                                   + len("  / "))
    else:
        bar_len = 30
    filled_len = int(round(bar_len * current_step / float(last_step)))
    bar = '=' * filled_len + '.' * (bar_len - filled_len)

    bar = f"{text}[{bar:s}] {current_step:d} / {last_step:d}"
    if current_step < last_step-1:
        # Erase to end of line and print
        sys.stdout.write("\033[K" + bar + "\r")
    else:
        sys.stdout.write("\033[K" + bar + "\r\n")

    sys.stdout.flush()


def accuracy(net, loader, DEVICE):
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total


def load_pretrain_weights(DEVICE, TRAIN_FROM_SCRATCH=True, RETRAIN=False, dataname='cifar10', 
                          train_loader=None, test_loader=None, 
                          checkpoints_folder="LIRA_checkpoints", SUFFIX=None, resume=False, SEED=42,
                          BREAKs=None, breakcount=0, GLIR=False):
    """
    directly download weights of a model trained exclusively on the retain set
    """
    if RETRAIN:
        local_path = f"retrain_weights_resnet18_{dataname}_{SUFFIX}_seed_{SEED}.pth" if SUFFIX else f"retrain_weights_resnet18_{dataname}_seed_{SEED}.pth"
    else:
        local_path = f"weights_resnet18_{dataname}.pth"

    if dataname == 'cifar10' or dataname == 'cinic10' or dataname == 'incremental':
        num_classes = 10
    elif dataname == 'cifar100':
        num_classes = 100

    break_epoch = breakcount * int(201/BREAKs) if BREAKs is not None else 0

    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)
    if not os.path.exists(checkpoints_folder+ '/unle_models/'):
        os.makedirs(checkpoints_folder+ '/unle_models/')

    if RETRAIN and SUFFIX:
        model_path = checkpoints_folder + '/unle_models/' + local_path 
    elif RETRAIN:
        model_path = checkpoints_folder + '/' + local_path
    else:
        model_path = checkpoints_folder + '/' + local_path

    print(f"model_path: {model_path}")
    if (not os.path.exists(model_path) or GLIR or (BREAKs is not None and not os.path.exists(model_path.split('.pth')[0] + f"_break_{break_epoch}.pth"))) and TRAIN_FROM_SCRATCH:
        # print the training time
        if dataname == 'incremental':
            init_model_path = model_path.replace('incremental', 'cifar10')
            weights_pretrained = torch.load(
                init_model_path, map_location=torch.device(DEVICE))
            # load model with pre-trained weights
            model = ResNet18(num_classes=num_classes)
            model.load_state_dict(weights_pretrained)
            model.to(DEVICE)
            model.eval()
            print('==> Loading pre-trained weights..on cifar10')
        else:
            model = ResNet18(num_classes=num_classes)
            model.to(DEVICE)

        start_time = time.time()
        # model = resnet18(weights=None, num_classes=num_classes).to(DEVICE)
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
        epochs = 201
        if GLIR:
            tot_grad_list = []
            tot_params_list = []
            epochs = 5
            
        for epoch in range(epochs):
            model.train()
            if GLIR and epoch < 5:
                step_grad_list = []
                step_param_list = []
                last_four_params = list(model.parameters())[-4:]
                # Initialize gradient accumulation
                accumulated_gradients = [torch.zeros_like(param) for param in last_four_params]
                num_batches = len(train_loader)

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                # print("loss: ", loss)
                if GLIR and epoch < 5:
                    # Accumulate gradients
                    for i, param in enumerate(last_four_params):
                        if param.grad is not None:
                            accumulated_gradients[i] += param.grad.detach()
                optimizer.step()
            scheduler.step()

            if GLIR and epoch < 5:
                state_dict = model.state_dict()
                step_param_list.append(state_dict_to_cpu(copy.deepcopy(state_dict)))
                # Perform step
                avg_gradients = [g / num_batches for g in accumulated_gradients]
                grad_cpu = [g.cpu().clone() for g in avg_gradients]
                step_grad_list.append(list_to_cpu(grad_cpu))
                tot_grad_list.append(step_grad_list)
                tot_params_list.append(step_param_list)

            model.eval()
            train_acc = accuracy(model, train_loader, DEVICE)
            acc = accuracy(model, test_loader, DEVICE)
            print(
                f"Epoch {epoch+1} | Training accuracy: {train_acc} | Validation accuracy: {acc} | Train time: {time.time() - start_time}")

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
                torch.save(state['net'], model_path.split('.pth')[0] + f"_break_{epoch}.pth")



        if GLIR:
            res_dict = {"stepwise_params": tot_params_list, 
                        "stepwise_grads": tot_grad_list}
            torch.save(res_dict, model_path.split('.pth')[0] + f"_grads_params_GLIR.pth")
        else:
            # save model weights
            model.to('cpu')
            # state['net'].to('cpu')
            torch.save(state['net'], model_path)
        # clean up
        del model, optimizer, scheduler, criterion

    if BREAKs is not None:
        weights_pretrained = torch.load(
            model_path.split('.pth')[0] + f"_break_{break_epoch}.pth", map_location=torch.device(DEVICE))
        return weights_pretrained, model_path
    elif GLIR:
        return None
    else:
        weights_pretrained = torch.load(
            model_path, map_location=torch.device(DEVICE))

        return weights_pretrained, model_path

def load_pretrain_weights_breakpoint(DEVICE, dataname='cifar10', 
                                     train_loader=None, retain_loader=None, test_loader=None, 
                                     breakpoint=-1, checkpoints_folder="LIRA_checkpoints/newseed/", SUFFIX=None, resume=False):
    """
    directly download weights of a model trained exclusively on the retain set
    """
    if breakpoint == -1:
        local_path = f"weights_resnet18_{dataname}_break_{breakpoint}.pth"
    else:
        local_path = f"weights_resnet18_{dataname}_{SUFFIX}_break_{breakpoint}.pth" if SUFFIX else f"weights_resnet18_{dataname}_break_{breakpoint}.pth"
    
    if dataname == 'cifar10':
        num_classes = 10
    elif dataname == 'cifar100':
        num_classes = 100

    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)
    model_path = checkpoints_folder + '/' + local_path
    print(f"model_path: {model_path}")
    if not os.path.exists(model_path):
        # print the training time
        start_time = time.time()
        model = ResNet18(num_classes=num_classes)
        # model = resnet18(weights=None, num_classes=num_classes).to(DEVICE)
        model.to(DEVICE)
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
        breakpoint = 201 if breakpoint == -1 else breakpoint
        for epoch in range(201):
            if epoch < breakpoint:
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
                print(
                    f"Epoch {epoch+1} | Validation accuracy: {acc} | Train time: {time.time() - start_time}")

                if acc > best_acc:
                    print('Saving..')
                    state = {
                        'net': model.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                    }
                    torch.save(state, checkpoints_folder + '/ckpt.pth')
                    best_acc = acc
            else:
                model.train()
                for inputs, targets in retain_loader:
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

                if epoch == breakpoint:
                    print("Breakpoint reached, switch to training on the retained set")

                print(
                    f"Epoch {epoch+1} | Validation accuracy: {acc} | Train time: {time.time() - start_time}")

                if acc > best_acc:
                    print('Saving..')
                    state = {
                        'net': model.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                    }
                    best_acc = acc
        # save model weights
        model.to('cpu')
        # state['net'].to('cpu')
        torch.save(state['net'], model_path)
        # clean up
        del model, optimizer, scheduler, criterion

    weights_pretrained = torch.load(
        model_path, map_location=torch.device(DEVICE))

    return weights_pretrained, model_path

def get_model_init(model, device='cpu', seed=1):
    manual_seed(seed)
    # print("model: ", model)
    # get the model final layer's name
    # get the model final layer's number of classes
    num_classes = list(model.named_children())[-1][1].out_features

    arch = model._get_name()
    # print("arch: ", arch)
    # print("num_classes: ", num_classes)
    model_init = get_model(arch, num_classes=num_classes).to(device)
    # print("model_init: ", model_init)
    # print("model_init.parameters: ", model_init.parameters)
    # model_init.load_state_dict(torch.load(resume))
    for p in model_init.parameters():
        p.data0 = p.data.clone()
    # print("p.data0: ", p.data0)
    # print("p.data: ", p.data)
    return model_init


def get_model(arch, num_classes, filters_percentage=1):
    print(f'=> Building model..{arch}')
    if arch == 'ResNet':
        model = ResNet18(num_classes=num_classes)
    else:
        raise NotImplementedError
    return model

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy_l(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(data_loader, model, criterion, optimizer, DEVICE):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for (inputs, targets) in data_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # measure accuracy and record loss
        prec1, prec5 = accuracy_l(
            outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (losses.avg, top1.avg)

def train_classifier(train_loader, dataname, test_loader, TYPE, DEVICE, 
                     checkpoints_folder='LIRA_checkpoints', SUFFIX=None, 
                     PRIVACY=False, SEED=42, BREAKs=None, breakcount=0,
                     GLIR=False, ReSEED=False):
    if TYPE == 'ori':
        model_path = checkpoints_folder + '/weights_{}.pth'.format(dataname) if not PRIVACY else checkpoints_folder + '/weights_{}_privacy.pth'.format(dataname)
        if ReSEED:
            model_path = model_path.split('.pth')[0] + f"_seed_{SEED}.pth"    
            
    elif TYPE == 'retrain':
        model_path = checkpoints_folder + f'/unle_models/retrain_weights_{dataname}_{SUFFIX}_seed_{SEED}.pth' 
    
    if dataname == 'purchase':
        model = PurchaseClassifier()
        lr = 0.01
        epochs = 100
    
    if TYPE == 'unlearn': # directly load the model trained on the original set
        model_path = checkpoints_folder + '/weights_{}.pth'.format(dataname)  
        # model_path = model_path.split('.pth')[0] + f"_seed_{SEED}.pth"
        state_dict = torch.load(model_path)
        new_state_dict = {
                k.replace('_module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        model = model.to(DEVICE)
        return model
    
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    break_epoch = breakcount * int(epochs/BREAKs) if BREAKs is not None else 0

    print(f"model_path: {model_path}")
    if not os.path.exists(model_path) or GLIR or (BREAKs is not None and not os.path.exists(model_path.split('.pth')[0] + f"_break_{break_epoch}.pth")):
        criterion = nn.CrossEntropyLoss()
        model = model.to(DEVICE)
        if not PRIVACY:
            start_time = time.time()
            optimizer = optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

            best_acc = 0
            if GLIR:
                epochs = 5
                tot_grad_list = []
                tot_params_list = []
                
            for epoch in range(epochs):
                model.train()
                if GLIR and epoch < 5:
                    step_grad_list = []
                    step_param_list = []
                    last_four_params = list(model.parameters())[-4:]
                    # Initialize gradient accumulation
                    accumulated_gradients = [torch.zeros_like(param) for param in last_four_params]
                    num_batches = len(train_loader)

                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    # print("loss: ", loss)
                    if GLIR and epoch < 5:
                        # Accumulate gradients
                        for i, param in enumerate(last_four_params):
                            if param.grad is not None:
                                accumulated_gradients[i] += param.grad.detach()

                    optimizer.step()
                scheduler.step()

                if GLIR and epoch < 5:
                    state_dict = model.state_dict()
                    step_param_list.append(state_dict_to_cpu(copy.deepcopy(state_dict)))
                    # Perform step
                    avg_gradients = [g / num_batches for g in accumulated_gradients]
                    grad_cpu = [g.cpu().clone() for g in avg_gradients]
                    step_grad_list.append(list_to_cpu(grad_cpu))
                    tot_grad_list.append(step_grad_list)
                    tot_params_list.append(step_param_list)

                model.eval()
                acc = accuracy(model, test_loader, DEVICE)
                print(
                    f"Epoch {epoch+1} | Validation accuracy: {acc} | Train time: {time.time() - start_time}")

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
                    torch.save(state['net'], model_path.split('.pth')[0] + f"_break_{epoch}.pth")
                    print(f"internal model saved as: {model_path.split('.pth')[0] + f'_break_{epoch}.pth'}")

            if GLIR:
                res_dict = {"stepwise_params": tot_params_list, 
                            "stepwise_grads": tot_grad_list}
                torch.save(res_dict, model_path.split('.pth')[0] + f"_grads_params_GLIR.pth")
            else:
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
        model.load_state_dict(torch.load(model_path.split('.pth')[0] + f"_break_{break_epoch}.pth", map_location=torch.device(DEVICE)))
        return model
    elif GLIR:
        return None
    else:
        return model
    
def train_classifier_breakpoint(dataname, train_loader, retain_loader=None, test_loader=None, 
                                breakpoint=-1, DEVICE='cpu', checkpoints_folder='LIRA_checkpoints/newseed/', SUFFIX=None, PRIVACY=False):

    if breakpoint == -1:
        local_path = f"weights_{dataname}_break_{breakpoint}.pth"
    else:
        local_path = f"weights_{dataname}_{SUFFIX}_break_{breakpoint}.pth" if SUFFIX else f"weights_{dataname}_break_{breakpoint}.pth" 

    if dataname == 'purchase':
        model = PurchaseClassifier()
    
    lr = 0.01
    
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)
    model_path = checkpoints_folder + '/' + local_path

    print(f"model_path: {model_path}")
    if not os.path.exists(model_path):
        criterion = nn.CrossEntropyLoss()
        model = model.to(DEVICE)
        start_time = time.time()
        optimizer = optim.SGD(model.parameters(), lr=lr,
                            momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        best_acc = 0
        breakpoint = 201 if breakpoint == -1 else breakpoint
        for epoch in range(201):
            if epoch < breakpoint:
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
                print(
                    f"Epoch {epoch+1} | Validation accuracy: {acc} | Train time: {time.time() - start_time}")

                if acc > best_acc:
                    print('Saving..')
                    state = {
                        'net': model.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                    }
                    best_acc = acc
            else:
                model.train()
                for inputs, targets in retain_loader:
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
                
                if epoch == breakpoint:
                    print("Breakpoint reached, switch to training on the retained set")

                print(
                    f"Epoch {epoch+1} | Validation accuracy: {acc} | Train time: {time.time() - start_time}")

                if acc > best_acc:
                    print('Saving..')
                    state = {
                        'net': model.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                    }
                    best_acc = acc

        # save model weights
        model.to('cpu')
        # state['net'].to('cpu')
        torch.save(state['net'], model_path)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
        # clean up
        del optimizer, scheduler, criterion

    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    
    return model

def get_model(arch, num_classes, filters_percentage=1):
    print(f'=> Building model..{arch}')
    if arch == 'ResNet':
        model = ResNet18(num_classes=num_classes)
    elif arch == 'PurchaseClassifier':
        model = PurchaseClassifier()
    else:
        raise NotImplementedError
    return model

def state_dict_to_cpu(sdict, device = "cpu"):
    for k, v in sdict.items():
        sdict[k] = sdict[k].to(device)
    return sdict


def list_to_cpu(slist, device = "cpu"):
    for i in range(len(slist)):
        slist[i] = slist[i].to(device)
    return slist


def train_attack_model(args=None, shadow_set_len=20000, model_type='LR', 
                       model_numbers=16, model_list=None, shadow_path=None, DEVICE=torch.device("cpu"), 
                       train_transforms=None, test_transforms=None, batch_size=1024, shuffle=True, 
                       num_workers=2, ratio=0.8, arch=None):
    '''
    train attack model on nonmem set
    '''
    if model_list is not None:
        model_numbers = len(model_list)
    else:
        model_list = list(range(model_numbers))
        
    save_path = shadow_path + f'/attack_model_unleak_{model_numbers}_'+'_'.join(map(str, model_list))+'.pkl'
    if arch is not None:
        save_path = shadow_path + f'/attack_model_unleak_{model_numbers}_'+'_'.join(map(str, model_list))+'_{arch}.pkl'

    if len(save_path) > 200:
        save_path = save_path.split('.pkl')[0][:200]+'.pkl'
        
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            attack_model = pickle.load(f)
        print('Load trained attack model')
        return attack_model

    # attack model
    dataname = shadow_path.split('/')[-1]
    if dataname == 'cifar10' or dataname == 'cinic10' or dataname == 'cifar100' or dataname == 'texas':
        shadow_set_len = 20000
    elif dataname == 'location':
        shadow_set_len = 1000
    elif dataname == 'purchase':
        shadow_set_len = 40000

    attack_model_mlp = MLPClassifier(early_stopping=True, learning_rate_init=0.01, max_iter=1000)

    X_shadows, y_shadows = [], []
    X_shadows_lira = []
    if shadow_path is not None:
        print('Load pre-trained shadow logits for UnLeak-Attack')
        load_path = shadow_path + f'/shadow_origin_model.pth'
        load_path_logit = shadow_path + f'/shadow_origin_model_logit.pth'
        load_path_label = shadow_path + f'/shadow_origin_model_label.pth'

        shadow_origin_logit = torch.load(load_path_logit, map_location=torch.device('cpu'))
        shadow_origin_label = torch.load(load_path_label, map_location=torch.device('cpu'))
        RNG = torch.Generator()
        RNG.manual_seed(42)
        for shadow_idx in model_list:
            load_path = shadow_path + f'/shadow_model_{shadow_idx}.pth'
            load_path_logit_train = shadow_path + f'/shadow_model_{shadow_idx}_logit_train.pth'
            load_path_label_train = shadow_path + f'/shadow_model_{shadow_idx}_label_train.pth'
            load_path_logit_test = shadow_path + f'/shadow_model_{shadow_idx}_logit_test.pth'
            load_path_label_test = shadow_path + f'/shadow_model_{shadow_idx}_label_test.pth'
            shadow_local_logit_train = torch.load(load_path_logit_train, map_location=torch.device('cpu'))
            shadow_local_label_train = torch.load(load_path_label_train, map_location=torch.device('cpu'))
            shadow_local_logit_test = torch.load(load_path_logit_test, map_location=torch.device('cpu'))
            shadow_local_label_test = torch.load(load_path_label_test, map_location=torch.device('cpu'))
            FIX_TRAIN_SIZE = 200
            TRAIN_SHADOW_SIZE = int(shadow_set_len*ratio)-FIX_TRAIN_SIZE
            # SPLIT_LIST = [TRAIN_SHADOW_SIZE, shadow_set_len-TRAIN_SHADOW_SIZE]
            RNG.manual_seed(RNG.get_state()[0].item() + shadow_idx)

            # get the indices of the new train set in the original dataset
            shuffled_idx = torch.randperm(
                shadow_set_len-FIX_TRAIN_SIZE, generator=RNG)
            shadow_train_idx = shuffled_idx[:TRAIN_SHADOW_SIZE]
            # merge with fixed train set
            shadow_train_idx = torch.cat((shadow_train_idx, torch.tensor(
                range((shadow_set_len-FIX_TRAIN_SIZE), shadow_set_len))), 0)
            shadow_test_idx = shuffled_idx[TRAIN_SHADOW_SIZE:]

            target_in_probs = torch.softmax(shadow_origin_logit[shadow_test_idx,:], dim=1) 
            target_out_probs = torch.softmax(shadow_local_logit_test, dim=1) 
            neg_old_prob = torch.softmax(shadow_origin_logit[shadow_train_idx,:], dim=1)[:len(shadow_test_idx),:]
            neg_new_prob = torch.softmax(shadow_local_logit_train, dim=1)[:len(shadow_test_idx),:] 

            target_pos_feature = construct_leak_feature(
                target_in_probs, target_out_probs)

            other_neg_feature = construct_leak_feature(
                neg_old_prob, neg_new_prob)

            target_pos_labels = np.ones((target_pos_feature.shape[0]))
            other_neg_labels = np.zeros((other_neg_feature.shape[0]))

            X_shadow = np.concatenate([other_neg_feature, target_pos_feature])
            y_shadow = np.concatenate([other_neg_labels, target_pos_labels])
            X_shadows.append(X_shadow)
            y_shadows.append(y_shadow)

            # target_pos_feature_lira = construct_leak_feature(
            #     logit_scaling(target_in_probs), logit_scaling(target_out_probs))
            # other_neg_feature_lira = construct_leak_feature(
            #     logit_scaling(neg_old_prob), logit_scaling(neg_new_prob))

            # X_shadow_lira = np.concatenate(
            #     [target_pos_feature_lira, other_neg_feature_lira])
            # X_shadows_lira.append(X_shadow_lira)

    else:
        print('Train shadow models to get logits for UnLeak-Attack')
        features_list = []
        labels_list = []
        _, _, _, _, _, shadow_set,_,_,_ = get_data_loaders(args.dataname, batch_size=args.batch_size, num_workers=args.num_workers, forget_size=args.forget_size, shuffle=args.shuffle)
        nonmem_loader = torch.utils.data.DataLoader(
            shadow_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        
        for batch in nonmem_loader:
            features, labels = batch
            features_np = features.cpu().numpy()
            labels_np = labels.cpu().numpy()
            if features_np.ndim > 2:
                features_np = features_np.reshape(features_np.shape[0], -1)

            features_list.append(features_np)
            labels_list.append(labels_np)
        X_all = np.concatenate(features_list, axis=0)
        y_all = np.concatenate(labels_list, axis=0)

        for i in model_list:
            # shadow models: mimic the origin model and unlearned model
            origin_model = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000))
            unlearn_model = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000))
            train_indices = np.random.choice(
                X_all.shape[0], int(X_all.shape[0]*0.9), replace=False)
            test_indices = np.setdiff1d(np.arange(X_all.shape[0]), train_indices)
            # generate target indices from train_indices
            unlearned_indices = np.random.choice(train_indices, int(
                train_indices.shape[0]*0.1), replace=False)
            retained_indices = np.setdiff1d(train_indices, unlearned_indices)

            train_features = X_all[train_indices] # samples in original model
            train_labels = y_all[train_indices]
            test_features = X_all[test_indices]
            test_labels = y_all[test_indices]
            unlearned_features = X_all[unlearned_indices] # kept features
            unlearned_labels = y_all[unlearned_indices]
            retained_features = X_all[retained_indices] # kept samples in unlearned model
            retained_labels = y_all[retained_indices]
            origin_model.fit(train_features, train_labels)
            unlearn_model.fit(retained_features, retained_labels)
            # collect training data for attack model
            target_in_probs = origin_model.predict_proba(unlearned_features)
            target_out_probs = unlearn_model.predict_proba(unlearned_features)
            neg_old_prob = origin_model.predict_proba(retained_features)
            neg_new_prob = unlearn_model.predict_proba(retained_features)

            target_pos_feature = construct_leak_feature(torch.from_numpy(
                target_in_probs), torch.from_numpy(target_out_probs))
            other_neg_feature = construct_leak_feature(
                torch.from_numpy(neg_old_prob), torch.from_numpy(neg_new_prob))

            other_neg_labels = np.zeros((other_neg_feature.shape[0]))
            target_pos_labels = np.ones((target_pos_feature.shape[0]))

            X_shadow = np.concatenate([other_neg_feature, target_pos_feature])
            y_shadow = np.concatenate([other_neg_labels, target_pos_labels])

            X_shadows.append(X_shadow)
            y_shadows.append(y_shadow)

    X_shadows = np.concatenate(X_shadows, axis=0)
    y_shadows = np.concatenate(y_shadows, axis=0)
    X_shadows = np.where(np.isinf(X_shadows), 0, X_shadows)
    start_time  = time.time()

    attack_model_mlp.fit(X_shadows, y_shadows)
    print(f'{attack_model_mlp.classes_}')
    train_acc = attack_model_mlp.score(X_shadows, y_shadows)
    train_pred = attack_model_mlp.predict(X_shadows)
    train_acc_self = np.mean(train_pred == y_shadows)
    sample_prob = attack_model_mlp.predict_proba(X_shadows)[-10:,1]
    print(f'train_acc: {train_acc}, train_acc_self: {train_acc_self}')
    print(f'sample_prob: {sample_prob}, sample_label: {y_shadows[-10:]}')
    print('attack Training time:', time.time()-start_time)
    # save attack_model
    with open(save_path,'wb') as f:
        pickle.dump(attack_model_mlp,f)

    return attack_model_mlp

def construct_leak_feature(target_in_probs, target_out_probs):
    # construct feature vector for target set ["direct_diff", "sorted_diff", 'direct_concat', 'sorted_concat', 'l2_distance', 'basic_mia']
    # cor probs, use top-3 probs, and the  the others use average of `1-sum(posterior values)
    def topk_posterior_probs(probs, topk=3):
        num_classes = probs.shape[1]
        _, top_indices = torch.topk(probs, k=topk, dim=1, largest=True)
        mask = torch.ones_like(probs, dtype=torch.bool)
        mask.scatter_(1, top_indices, False)
        result = torch.zeros_like(probs)
        result.scatter_(1, top_indices, torch.gather(probs, 1, top_indices))
        top_k_sum = torch.sum(torch.gather(probs, 1, top_indices), dim=1, keepdim=True)
        remaining_prob = (1 - top_k_sum) / (num_classes - 3)
        result[mask] = remaining_prob.repeat_interleave(num_classes - topk)
        return result
    
    target_in_probs = topk_posterior_probs(target_in_probs)
    target_out_probs = topk_posterior_probs(target_out_probs)
    
    # unsorted feature
    direct_diff = target_in_probs - target_out_probs
    l2_distance = torch.norm(direct_diff, dim=1).unsqueeze(1)
    direct_concat = torch.cat([target_in_probs, target_out_probs], dim=1)
    basic_mia = target_in_probs
    # sorted feature
    sort_indices = torch.argsort(target_in_probs, dim=1, descending=False)
    target_in_sort = torch.gather(target_in_probs, 1, sort_indices)
    target_out_sort = torch.gather(target_out_probs, 1, sort_indices)

    sorted_diff = target_in_sort - target_out_sort
    sorted_concat = torch.cat([target_in_sort, target_out_sort], dim=1)

    leak_feature = torch.cat(
        [direct_diff, sorted_diff, direct_concat, sorted_concat, l2_distance, basic_mia], dim=1)
    leak_feature = sorted_diff

    leak_feature = replace_inf_nan(leak_feature)
    return leak_feature
