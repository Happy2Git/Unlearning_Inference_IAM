import torch
import copy
import os
import psutil
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict, defaultdict
from ..utils import display_progress, manual_seed, get_model_init
from ..models.resnet import ResNet18


def fisher_forgetting(net, retain_loader, batch_size=32, REPEAT=1, alpha: float = 1e-9, Method: str = 'EWC', DEVICE: str = 'cpu'):
    """
    Here, we implement the diagnoal fisher estimate by a more efficient way.
    Reference:  Aditya Golatkar, Alessandro Achille, Stefano Soatto:
    Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks. CVPR 2020: 9301-9309
    """
    if Method == 'EWC':
        modelf = ewc_estimate_fisher(
            net, retain_loader, batch_size=batch_size, REPEAT=REPEAT, DEVICE=DEVICE)
    else:
        modelf = reference_fisher_init(net, retain_loader, DEVICE)

    # check the classes of retain_loader if there is a class to forget
    # if there is a class to forget, we set the mu and var of that class to 0
    num_classes = list(net.named_children())[-1][1].out_features

    existing_class = []
    for x, y in retain_loader:
        for i in y:
            if i not in existing_class:
                existing_class.append(i)

    class_to_forget = None
    if len(existing_class) < num_classes:
        class_to_forget = [i for i in range(
            num_classes) if i not in existing_class]

    # torch.manual_seed(seed)
    for i, p in enumerate(modelf.parameters()):
        mu, var = get_mean_var(p, num_classes, alpha=alpha,
                               class_to_forget=class_to_forget)
        mu, var = mu.to(DEVICE), var.to(DEVICE)
        # print(f"mu: {mu}, var: {var}")
        epsilon = 1e-30
        p.data = mu + var.sqrt() * torch.full_like(p.data, epsilon).normal_()
        # print(f"mu: {mu}, var: {var.sqrt() * torch.empty_like(p.data).normal_()}")

    return modelf


# ntk_fisher_forgetting: out of memory on large dataset (>10000)
def ntk_fisher_forgetting(model, retain_loader, forget_loader, alpha=1e-8, weight_decay=0.1, DSLIMIT: int = 600, DEVICE: str = 'cpu'):
    """
    Reference:  Aditya Golatkar, Alessandro Achille, Stefano Soatto:
    Forgetting Outside the Box: Scrubbing Deep Networks of Information Accessible 
    from Input-Output Observations. ECCV (29) 2020: 383-398
    """
    if not os.path.exists('NTK_data'):
        os.makedirs('NTK_data')
    net_scrubbed = copy.deepcopy(model)

    # ntk scrubbing
    model_init = get_model_init(model, DEVICE)
    direction, scale = ntk_scrub_vector(
        model, model_init, retain_loader, forget_loader, weight_decay, DSLIMIT=DSLIMIT)
    for k, p in net_scrubbed.named_parameters():
        p.data += (direction[k]*scale).to(DEVICE)
    # fisher forgetting
    net = fisher_forgetting(net_scrubbed, retain_loader,
                            alpha=alpha, DEVICE=DEVICE)
    net.eval()
    return net


def ewc_estimate_fisher(model_input, retain_loader, batch_size=32, REPEAT=1, DEVICE: str = 'cpu'):
    """
    The diagonal terms of the empirical Fisher information matrix can be approximated 
        by taking the average of the squared gradients over the data distribution.

    ref: https://github.com/kuc2477/pytorch-ewc/blob/master/model.py

    self suggestions: 
        - Use a large batch size to get more stable gradient estimates. 
            limit the number of samples used to compute Fisher (say 256), which is good for most use-cases.
        - Iterate through the full dataset at least 2-3 times to get smoother values. 
            The code goes through once.
    """
    # sample loglikelihoods from the dataset.
    modelf = copy.deepcopy(model_input)
    for p in modelf.parameters():
        p.data0 = copy.deepcopy(p.data.clone())

    retain_loader_b1 = torch.utils.data.DataLoader(
        retain_loader.dataset, batch_size=batch_size, shuffle=True)
    num_samples = len(retain_loader_b1.dataset)
    epsilon = 1e-30
    fisher_diagonals = [torch.full_like(p, epsilon, device='cpu') for p in modelf.parameters()]
   
    for repeat in range(REPEAT):
        for enu, (x, y) in enumerate(retain_loader_b1):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            loglikelihoods = F.log_softmax(modelf(x), dim=1)[
                range(x.shape[0]), y.data]
            # estimate the fisher information of the parameters.
            loglikelihoods = loglikelihoods.unbind()
            loglikelihood_grads = zip(*[torch.autograd.grad(
                l, modelf.parameters(),
                retain_graph=(i < len(loglikelihoods))
            ) for i, l in enumerate(loglikelihoods, 1)])
            loglikelihood_grads = [torch.stack(
                gs).detach() for gs in loglikelihood_grads]
            for i, g in enumerate(loglikelihood_grads):
                fisher_diagonals[i] += torch.clamp(
                    (g**2).sum(dim=0).cpu(), min=0)

            # # del loglikelihood_grads, loglikelihoods, x, y
            # # torch.cuda.empty_cache()
            # display_progress("fisher estimation", enu+len(retain_loader_b1)
            #                  * repeat, len(retain_loader_b1)*REPEAT)

    for i in range(len(fisher_diagonals)):
        fisher_diagonals[i] /= (num_samples*REPEAT)
    # let the param.grad2_acc be the fisher information.
    for p, f in zip(modelf.parameters(), fisher_diagonals):
        p.grad2_acc = f.detach().clone()

    return modelf


def reference_fisher_init(model_input, retain_loader, DEVICE):
    """
    Reference:  Aditya Golatkar, Alessandro Achille, Stefano Soatto:
    Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks. CVPR 2020: 9301-9309
    """
    modelf = copy.deepcopy(model_input)
    for p in modelf.parameters():
        p.data0 = copy.deepcopy(p.data.clone())

    modelf.eval()
    retain_loader_b1 = torch.utils.data.DataLoader(
        retain_loader.dataset, batch_size=1, shuffle=False)
    loss_fn = torch.nn.CrossEntropyLoss()

    for p in modelf.parameters():
        p.grad2_acc = 0

    for i, (data, orig_target) in enumerate(retain_loader_b1):
        data, orig_target = data.to(DEVICE), orig_target.to(DEVICE)
        output = modelf(data)
        prob = F.softmax(output, dim=-1).data

        for y in range(output.shape[1]):
            target = torch.full_like(orig_target,y)
            loss = loss_fn(output, target)
            modelf.zero_grad()
            loss.backward(retain_graph=True)
            for p in modelf.parameters():
                if p.requires_grad:
                    p.grad2_acc += torch.clamp(prob[:, y]
                                               * p.grad.data.pow(2), min=0)

        display_progress("Fisher init", i, len(
            retain_loader_b1.dataset), enabled=True)

    for p in modelf.parameters():
        p.grad2_acc /= len(retain_loader_b1)
    return modelf


def get_mean_var(p, num_classes, alpha=1e-9, class_to_forget=None):
    mu = copy.deepcopy(p.data0.clone())
    var = copy.deepcopy(1./torch.abs(p.grad2_acc+1e-8))
    var = var.clamp(max=1e3)
    if p.size(0) == num_classes:
        var = var.clamp(max=1e2)
    var = alpha * var

    if p.ndim > 1:
        var = var.mean(dim=1, keepdim=True).expand_as(p).clone()

    # class forgetting
    if p.size(0) == num_classes and class_to_forget is not None:
        mu[class_to_forget] = 0
        var[class_to_forget] = 0.0001

    if p.size(0) == num_classes:
        # Last layer
        var *= 10
    elif p.ndim == 1:
        # BatchNorm
        var *= 10
#         var*=1
    return mu, var


def ntk_scrub_vector(model, model_init, retain_loader, forget_loader, weight_decay=0.1, DSLIMIT=None):
    """
    If len(dataset) is 600, class_number is 10, then G.shape of ResNet18 is (600*10, 11181642), 
    the size of G is ~25G in float32 format, which is too large to store in memory.
    So we split the dataset into two parts, retain and forget.
    """
    dlen = len(retain_loader.dataset)+len(forget_loader.dataset)
    plen = sum([np.prod(list(p.shape)) for p in model.parameters()])
    num_classes = list(model_init.named_children())[-1][1].out_features
    if dlen * plen * num_classes > 1e10 and DSLIMIT is None:
        raise ValueError(
            "The size of G will be too large (> 40GB) to store in memory.")

    # calculate G_r
    G_r_cpt, f0_minus_y_r_cpt = delta_w_utils(copy.deepcopy(
        model), retain_loader, 'retain', DSLIMIT=int(DSLIMIT*len(retain_loader.dataset)/dlen))
    print("save the checkpoint of _cpt in {} and f0_minus_y_r in {}".format(
        G_r_cpt, f0_minus_y_r_cpt))

    G_f_cpt, f0_minus_y_f_cpt = delta_w_utils(copy.deepcopy(
        model), forget_loader, 'forget', DSLIMIT=int(DSLIMIT*len(forget_loader.dataset)/dlen))
    print("save the checkpoint of G_f in {} and f0_minus_y_f in {}".format(
        G_f_cpt, f0_minus_y_f_cpt))

    # load all checkpoints of G_r and G_f and concatenate them
    G = []
    for i in range(int(G_r_cpt.split('cpt')[-1].split('.')[0])):
        G.append(torch.load("NTK_data/G_list_retain_cpt{}.pth".format(i+1)))
    for i in range(int(G_f_cpt.split('cpt')[-1].split('.')[0])):
        G.append(torch.load("NTK_data/G_list_forget_cpt{}.pth".format(i+1)))
    G = torch.cat(G, dim=1)

    torch.save(G, "NTK_data/G.pth")
    del G
    # todo, use the checkpoint to save memory

    f0_minus_y = []
    for i in range(int(f0_minus_y_r_cpt.split('cpt')[-1].split('.')[0])):
        f0_minus_y.append(torch.load(
            "NTK_data/f0_minus_y_retain_cpt{}.pth".format(i+1)))
    for i in range(int(f0_minus_y_f_cpt.split('cpt')[-1].split('.')[0])):
        f0_minus_y.append(torch.load(
            "NTK_data/f0_minus_y_forget_cpt{}.pth".format(i+1)))
    f0_minus_y = torch.cat(f0_minus_y, dim=0)
    torch.save(f0_minus_y, "NTK_data/f0_minus_y.pth")

    G = torch.load("NTK_data/G.pth")
    print("G.shape: ", G.shape, "f0_minus_y.shape: ", f0_minus_y.shape)
    theta = G.T @ G + (len(retain_loader.dataset) +
                       len(forget_loader.dataset))*weight_decay*torch.eye(G.shape[1])
    torch.save(theta, "NTK_data/theta.pth")
    del G, f0_minus_y

    theta_inv = torch.inverse(theta)
    torch.save(theta_inv, "NTK_data/theta_inv.pth")
    del theta

    G = torch.load("NTK_data/G.pth")
    f0_minus_y = torch.load("NTK_data/f0_minus_y.pth")
    w_complete = - G @ (theta_inv @ f0_minus_y)

    torch.save(w_complete, "NTK_data/w_complete.pth")
    del G, f0_minus_y, theta_inv, w_complete

    G_r = []
    for i in range(int(G_r_cpt.split('cpt')[-1].split('.')[0])):
        G_r.append(torch.load("NTK_data/G_list_retain_cpt{}.pth".format(i+1)))
    G_r = torch.cat(G_r, dim=1)
    theta_r = G_r.T @ G_r + len(retain_loader.dataset) * \
        weight_decay*torch.eye(G_r.shape[1])
    torch.save(theta_r, "NTK_data/theta_r.pth")
    del G_r

    theta_r_inv = torch.inverse(theta_r)
    torch.save(theta_r_inv, "NTK_data/theta_r_inv.pth")
    del theta_r

    G_r = []
    for i in range(int(G_r_cpt.split('cpt')[-1].split('.')[0])):
        G_r.append(torch.load("NTK_data/G_list_retain_cpt{}.pth".format(i+1)))
    G_r = torch.cat(G_r, dim=1)

    f0_minus_y_r = []
    for i in range(int(f0_minus_y_r_cpt.split('cpt')[-1].split('.')[0])):
        f0_minus_y_r.append(torch.load(
            "NTK_data/f0_minus_y_retain_cpt{}.pth".format(i+1)))
    f0_minus_y_r = torch.cat(f0_minus_y_r, dim=0)

    w_retain = -G_r @ (theta_r_inv @ f0_minus_y_r)
    torch.save(w_retain, "NTK_data/w_retain.pth")
    del G_r, f0_minus_y_r, theta_r_inv, w_retain

    w_complete = torch.load("NTK_data/w_complete.pth")
    w_retain = torch.load("NTK_data/w_retain.pth")
    delta_w = (w_retain-w_complete).squeeze()
    scale = trapezium_trick(model, model_init, delta_w, w_retain)
    direction = get_delta_w_dict(delta_w, model)
    return direction, scale


def delta_w_utils(model_init, dataloader, name='retain', lossfn='ce', dataset='cifar10', DSLIMIT=None):
    model_init.eval()
    DEVICE = next(model_init.parameters()).device
    num_classes = list(model_init.named_children())[-1][1].out_features
    dataloader = torch.utils.data.DataLoader(
        dataloader.dataset, batch_size=1, shuffle=False)
    # check the memory usage and calculate DSLIMIT and decide whether to save checkpoints of G_list and f0_minus_y
    CPU_mem, GPU_mem = get_memory_usage()
    DSLIMIT = DSLIMIT if DSLIMIT is not None else len(dataloader.dataset)
    SIZE = DSLIMIT * sum([np.prod(list(p.shape))
                         for p in model_init.parameters()]) * num_classes
    redundancy = 1.1
    Task_mem = redundancy*SIZE*4/(2**30)
    checkpoints_num = int(np.ceil(Task_mem/CPU_mem['free']))
    DSLIMIT = int(DSLIMIT/checkpoints_num)
    # raise ValueError("CPU_mem: {:.0f} GB, GPU_mem: {:.0f} GB, Task_mem: {:.0f} GB, checkpoints_num: {:.0f}".format(CPU_mem['free'], GPU_mem['free'], Task_mem, checkpoints_num))

    G_list = []
    f0_minus_y = []
    # (tqdm(dataloader,leave=False)):
    # use tqdm
    for idx, (input, target) in enumerate(dataloader):
        display_progress("ntk vector estimation", idx, len(dataloader))

        if DSLIMIT is not None and idx > DSLIMIT*checkpoints_num:
            break
        # elif DSLIMIT is not None:
            # print("idx: {:.0f} with maxmium {:.0f} GB with limit {:.0f} and  size {:.0f}".format(
            #     idx, SIZE*4/(2**30), DSLIMIT*checkpoints_num, SIZE))

        input, target = input.to(next(model_init.parameters()).device), target.to(
            next(model_init.parameters()).device)
        if 'mnist' in dataset:
            input = input.view(input.shape[0], -1)
        target = target.cpu()
        output = model_init(input)
        for cls in range(num_classes):
            RETAIN_GRAPH = True if cls < num_classes-1 else False
            grads = torch.autograd.grad(
                output[0, cls], model_init.parameters(), retain_graph=RETAIN_GRAPH)
            G_list.append(
                torch.cat([g.view(-1).detach().cpu() for g in grads]))
        if lossfn == 'mse':
            p = output.cpu().detach().T
            # loss_hess = np.eye(len(p))
            target = 2*target-1
            f0_y_update = p-target
            f0_minus_y.append(f0_y_update)
        elif lossfn == 'ce':
            p = torch.nn.functional.softmax(output, dim=1).cpu().detach().T
            p[target] -= 1
            f0_y_update = p
            f0_minus_y.append(f0_y_update)

        if idx % DSLIMIT == 0 and idx != 0:
            # save checkpoints and clear the memory
            torch.save(torch.stack(G_list).T,
                       "NTK_data/G_list_{}_cpt{}.pth".format(name, idx//DSLIMIT))
            torch.save(torch.vstack(
                f0_minus_y), "NTK_data/f0_minus_y_{}_cpt{}.pth".format(name, idx//DSLIMIT))
            G_list = []
            f0_minus_y = []

    if len(G_list) != 0:
        print(len(G_list), len(f0_minus_y))
        torch.save(torch.stack(G_list).T,
                   "NTK_data/G_list_cpt{}.pth".format(idx//DSLIMIT+1))
        torch.save(torch.vstack(f0_minus_y),
                   "NTK_data/f0_minus_y__cpt{}.pth".format(idx//DSLIMIT+1))
        G_list = []
        f0_minus_y = []
        return 'NTK_data/G_list_cpt{}.pth'.format(idx//DSLIMIT+1), 'NTK_data/f0_minus_y__cpt{}.pth'.format(idx//DSLIMIT+1)
    else:
        return 'NTK_data/G_list_cpt{}.pth'.format(idx//DSLIMIT), 'NTK_data/f0_minus_y__cpt{}.pth'.format(idx//DSLIMIT)


def get_delta_w_dict(delta_w, model):
    # Give normalized delta_w
    delta_w_dict = OrderedDict()
    params_visited = 0
    for k, p in model.named_parameters():
        num_params = np.prod(list(p.shape))
        update_params = delta_w[params_visited:params_visited+num_params]
        delta_w_dict[k] = torch.Tensor(update_params).view_as(p)
        params_visited += num_params
    return delta_w_dict


def vectorize_params(model):
    param = []
    for p in model.parameters():
        param.append(p.data.view(-1).cpu())
    return torch.cat(param)


def trapezium_trick(model, model_init, delta_w, w_retain, VERBOSE=False):
    m_pred_error = vectorize_params(
        model)-vectorize_params(model_init)-w_retain.squeeze()
    print(
        f"Pred Error Norm: {torch.linalg.norm(delta_w)}") if VERBOSE else None

    inner = torch.inner(delta_w/torch.linalg.norm(delta_w),
                        m_pred_error/torch.linalg.norm(m_pred_error))
    print(f"Inner Product--: {inner}") if VERBOSE else None

    if inner < 0:
        angle = torch.arccos(inner)-torch.pi/2
        print(f"Angle----------:  {angle}") if VERBOSE else None

        predicted_norm = torch.linalg.norm(
            delta_w) + 2*torch.sin(angle)*torch.linalg.norm(m_pred_error)
        print(f"Pred Act Norm--:  {predicted_norm}") if VERBOSE else None
    else:
        angle = torch.arccos(inner)
        print(f"Angle----------:  {angle}") if VERBOSE else None

        predicted_norm = torch.linalg.norm(
            delta_w) + 2*torch.cos(angle)*torch.linalg.norm(m_pred_error)
        print(f"Pred Act Norm--:  {predicted_norm}") if VERBOSE else None

    predicted_scale = predicted_norm/torch.linalg.norm(delta_w)
    print(f"Predicted Scale:  {predicted_scale}") if VERBOSE else None
    return predicted_scale


def get_memory_usage():
    """Returns the memory usage of the current process in MB"""

    total = psutil.virtual_memory().total / (1024**3)
    available = psutil.virtual_memory().available / (1024**3)
    CPU_mem = {'total': total, 'free': available}

    total_mem, free_mem = torch.cuda.mem_get_info()
    GPU_mem = {'total': total_mem / (1024**3), 'free': free_mem / (1024**3)}
    return CPU_mem, GPU_mem
