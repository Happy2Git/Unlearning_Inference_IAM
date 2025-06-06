import torch
from tqdm import tqdm
import copy
import numpy as np
from torch.nn.utils import parameters_to_vector as p2v
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

def L_CODEC(net, forget_loader,  test_loader,  dataset_name: str = 'cifar10', epochs: int = 5, lr: float = 1e-2, n_perturbations=1000, l2_reg=1e-2, net_folder: str = 'LIRA_checkpoints/', DEVICE: str = 'cpu', VERBOSE: bool = False):
    """
    Reference:  Ronak Mehta, Sourav Pal, Vikas Singh, Sathya N. Ravi:
    Deep Unlearning via Randomized Conditionally Independent Hessians. CVPR 2022: 10412-10421
    """
    if dataset_name == 'cifar10' or dataset_name == 'cifar100':
        net_path = net_folder + 'weights_resnet18_' + dataset_name + '.pth'
    else:
        net_path = net_folder + 'weights_' + dataset_name + '.pth'

    net.eval()
    net_copy = copy.deepcopy(net)
    net = net.to(DEVICE)
    if VERBOSE:
        print('Total params: %.2fM' % (sum(p.numel()
                                           for p in net.parameters())/1000000.0))
    # net.load_state_dict(torch.load(net_path))
    criterion = torch.nn.CrossEntropyLoss()
    # print("test_loader: ", test_loader)
    optim = torch.optim.SGD(net.parameters(), lr=lr)
    epochs = 2

    # with torch.no_grad():
    #     val_loss, val_accuracy = do_epoch(net, test_loader, criterion, 0, 0, optim=None, device=DEVICE)
    #     print(f'Model Before: val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

    param_bank_0, grad_bank_0, val_loss_0, val_accuracy_0 = do_epoch(
        net, test_loader, criterion, 0, epochs, optim=optim, device=DEVICE, outString=net_path, VERBOSE=VERBOSE)
    param_bank_1, grad_bank_1, val_loss_1, val_accuracy_1 = do_epoch(
        net, test_loader, criterion, 1, epochs, optim=optim, device=DEVICE, outString=net_path, VERBOSE=VERBOSE)
    if VERBOSE:
        print(
            f'Model Before: val_loss={val_loss_1:.4f}, val_accuracy={val_accuracy_1:.4f}')

    # prev_val_acc = val_accuracy
    # prev_statedict_fname = net_path.replace('.pth', '_prevSD.pth')
    # torch.save(net.state_dict(), prev_statedict_fname)
    optim = torch.optim.SGD(net.parameters(), lr=lr)
    # net.load_state_dict(torch.load(net_path))
    updatedSD = inp_perturb(
        net_copy, forget_loader, criterion, epochs, optim, l2_reg=l2_reg,
        device=DEVICE, outString=net_path, param_bank_0=param_bank_0, grad_bank_0=grad_bank_0,
        param_bank_1=param_bank_1, grad_bank_1=grad_bank_1, 
        n_perturbations=n_perturbations, VERBOSE=VERBOSE)
    net_copy.load_state_dict(updatedSD)

    return net_copy


# for L_CODEC
def do_epoch(model, dataloader, criterion, epoch, nepochs, optim=None, device='cpu', outString='', compute_grads=False, retrain=False, VERBOSE=False):
    # saves last two epochs gradients for computing finite difference Hessian
    total_loss = 0
    total_accuracy = 0
    grad_bank = None
    nsamps = 0
    if optim is not None:
        model.train()
    else:
        model.eval()

    if compute_grads:
        total_gradnorm = 0
    for x, y_true in tqdm(dataloader, leave=False):
        x, y_true = x.to(device), y_true.to(device)
        with torch.no_grad() if optim is None else torch.enable_grad():  # Disable gradients for validation/testing
            y_pred = model(x)
            loss = criterion(y_pred, y_true)

        # for training
        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()
            # saving for full/old Hessian estimate for scrubbing
            if epoch >= nepochs-2 and not retrain:
                batch_gradbank, param_bank = getGradObjs(model)
                if grad_bank is None:
                    grad_bank = batch_gradbank
                else:
                    for key in grad_bank.keys():
                        grad_bank[key] += batch_gradbank[key]

        nsamps += len(y_true)
        total_loss += loss.item()
        total_accuracy += (y_pred.max(1)[1] == y_true).float().sum().item()
        del x, y_true, y_pred, loss
        torch.cuda.empty_cache()

    if optim is not None:
        if epoch >= nepochs-2 and not retrain:
            for key in grad_bank.keys():
                grad_bank[key] = grad_bank[key]/nsamps
            # if VERBOSE:
            #     print(f'saving params/gradients at epoch {epoch}...')
            #     print("outString: ", outString)
            # torch.save(param_bank, outString.replace(
            #     '.pth', f'_epoch_{epoch}_params.pth'))
            # torch.save(grad_bank, outString.replace(
            #     '.pth', f'_epoch_{epoch}_grads.pth'))

    mean_loss = total_loss / len(dataloader)
    mean_accuracy = total_accuracy / nsamps
    torch.cuda.empty_cache()

    if compute_grads:
        mean_gradnorm = total_gradnorm / len(dataloader)
        return param_bank, grad_bank, mean_loss, mean_accuracy, mean_gradnorm
    else:
        return param_bank, grad_bank, mean_loss, mean_accuracy


class ActivationsHook(torch.nn.Module):

    def __init__(self, model):
        super(ActivationsHook, self).__init__()
        self.model = model
        self.model.eval()

        self.layers = []
        self.activations = []
        self.hooks = []

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                self.hooks.append(m.register_forward_hook(self.hook))
                self.layers.append(m)
            elif isinstance(m, torch.nn.Conv2d):
                self.hooks.append(m.register_forward_hook(self.convhook))
                self.layers.append(m)

    def getLayers(self):
        return self.layers

    def hook(self, module, input, output):

        # for batch size > 1
        output = output.mean(dim=[0])
        self.activations.append(output)

    def convhook(self, module, input, output):

        # for batch size > 1, and pixels (choose filters)
        flat = output.mean(dim=[0, 2, 3])
        self.activations.append(flat)

    def getActivations(self, x):
        self.activations = []
        output = self.model(x)
        return self.activations, output

    def clearHooks(self):
        for x in self.hooks:
            x.remove()

    def __del__(self):
        self.clearHooks()


def inp_perturb(model, data_loader, criterion, epochs,
                optim, device, outString, param_bank_0, grad_bank_0,
                param_bank_1, grad_bank_1, n_perturbations=1000,
                l2_reg=0.01, delta=0.01, epsilon=0.1,
                is_nlp=False, VERBOSE=False):
    
    hessian_device = 'cpu'
    x, y_true = next(iter(data_loader))
    x = x.to(device)
    y_true = y_true.to(device)

    model_copy = copy.deepcopy(model)

    myActs = ActivationsHook(model)

    torchLayers = myActs.getLayers()

    activations = []
    layers = None  # same for all, filled in by loop below
    losses = []

    model.eval()
    if VERBOSE:
        print("Starting input perturbation")
    for m in range(n_perturbations):
        tmpdata = x + (0.1)*torch.randn(x.shape).to(device)
        acts, out = myActs.getActivations(tmpdata.to(device))
        loss = criterion(out, y_true)
        vec_acts = p2v(acts)
        activations.append(vec_acts.detach())
        losses.append(loss.detach())

    acts = torch.vstack(activations)
    losses = torch.Tensor(losses).to(device)
    # descructor is not called on return for this
    # call it manually
    myActs.clearHooks()
    FOCIType = 'full'
    if VERBOSE:
        print('Running FOCI...')
    if FOCIType == 'full':
        if VERBOSE:
            print('Running full FOCI...')
        selectedActs, scores = foci(
            acts, losses, device, earlyStop=True, VERBOSE=False)
    # elif FOCIType == 'cheap':
    #     print('Running cheap FOCI...')
    #     selectedActs, scores = cheap_foci(acts, losses)

    # create mask for update
    slices_to_update = reverseLinearIndexingToLayers(selectedActs, torchLayers)
    if VERBOSE:
        print('Selected model blocks to update:')
        print(slices_to_update)

    ############ Sample Forward Pass ########
    model.train()
    model = DisableBatchNorm(model)
    total_loss = 0
    total_accuracy = 0
    y_pred = model(x)
    sample_loss_before = criterion(y_pred, y_true)
    if VERBOSE:
        print('Sample Loss Before: ', sample_loss_before)

    # Sample Gradient
    optim.zero_grad()
    sample_loss_before.backward()

    fullprevgradnorm = gradNorm(model)
    if VERBOSE:
        print('Sample Gradnorm Before: ', fullprevgradnorm)

    sampGrad1, _ = getGradObjs(model)
    vectGrad1, vectParams1, reverseIdxDict = getVectorizedGrad(
        sampGrad1, model, slices_to_update, hessian_device)
    model.zero_grad()
    train_epochs = epochs
    order = 'Hessian'
    approxType = 'FD'
    if order == 'Hessian':
        delwtm0, vectP0, _ = getVectorizedGrad(
            grad_bank_0, model, slices_to_update, hessian_device, paramlist=param_bank_0)
        
        delwtm1, vectP1, _ = getVectorizedGrad(
            grad_bank_1, model, slices_to_update, hessian_device, paramlist=param_bank_1)
        
        oldHessian = getHessian(
            delwtm0, delwtm1, approxType, w1=vectP0, w2=vectP1, hessian_device=hessian_device)
        # sample hessian
        model_copy.train()
        model_copy = DisableBatchNorm(model_copy)

        # for finite diff use a small learning rate
        # default adam is 0.001/1e-3, so use it here
        optim_copy = torch.optim.SGD(model_copy.parameters(), lr=1e-3)

        y_pred = model_copy(x)
        loss = criterion(y_pred, y_true)
        optim_copy.zero_grad()
        loss.backward()

        # step to get model at next point, compute gradients
        optim_copy.step()

        y_pred = model_copy(x)
        loss = criterion(y_pred, y_true)
        optim_copy.zero_grad()
        loss.backward()
        if VERBOSE:
            print('Sample Loss after Step for Hessian: ', loss)

        sampGrad2, _ = getGradObjs(model_copy)
        vectGrad2, vectParams2, _ = getVectorizedGrad(
            sampGrad1, model_copy, slices_to_update, device)
        sampleHessian = getHessian(vectGrad1, vectGrad2, approxType,
                                   w1=vectParams1, w2=vectParams2, hessian_device=hessian_device)
        del model_copy
        HessType = 'Sekhari'

        if HessType == 'Sekhari':
            # Sekhari unlearning update
            n = len(data_loader.dataset)
            combinedHessian = (
                1/(n-1))*(n*oldHessian.to(hessian_device) - sampleHessian.to(hessian_device))

            updatedParams = NewtonScrubStep(
                vectParams1, vectGrad1, combinedHessian, n, l2lambda=l2_reg, hessian_device=hessian_device)
            del vectParams1, vectGrad1, vectParams2, vectGrad2, sampleHessian, combinedHessian
            updatedParams = NoisyReturn(
                updatedParams, nsamps=n, m=1, lamb=l2_reg, epsilon=epsilon, delta=delta, device=hessian_device)

        elif HessType == 'CR':
            updatedParams = CR_NaiveNewton(
                vectParams1, vectGrad1, sampleHessian, l2lambda=l2_reg, hessian_device=hessian_device)

    with torch.no_grad():
        updatedParams = updatedParams.to(device)
        updateModelParams(updatedParams, reverseIdxDict, model)
    y_pred = model(x)
    loss2 = criterion(y_pred, y_true)
    if VERBOSE:
        print('Sample Loss After: ', loss2)
    optim.zero_grad()
    loss2.backward()

    fullscrubbedgradnorm = gradNorm(model)
    if VERBOSE:
        print('Sample Gradnorm After: ', fullscrubbedgradnorm)

    model.zero_grad()
    foci_val = 0

    return model.state_dict()


def gradNorm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2

    total_norm = total_norm ** (1. / 2)
    return total_norm


def getGradObjs(model):
    grad_objs = {}
    param_objs = {}
    for module in model.modules():
        # for module in model.modules():
        for (name, param) in module.named_parameters():
            grad_objs[(str(module), name)] = param.grad
            param_objs[(str(module), name)] = param.data

    return grad_objs, param_objs


def foci(X, Y, device, earlyStop=True, VERBOSE=False):
    p = X.shape[1]

    indeps = np.empty((p, 1))
    maxval = -100
    maxind = None
    for i in range(p):
        tmp = codec2(X[:, i], Y, device)
        if tmp > maxval:
            maxval = tmp
            maxind = i

    assert (maxval > -100)
    all_inds = np.arange(p)

    deplist = [maxind]
    depset = set(deplist)

    indepset = set(all_inds).difference(depset)
    indeplist = list(indepset)

    ordering = [maxind]
    codecVals = [maxval]
    if VERBOSE:
        print("p: ", p)
    for k in range(p-1):
        if VERBOSE:
            print("k: ", k)
        assert (list(depset.intersection(indepset)) == [])
        assert (len(list(depset.union(indepset))) == p)

        if VERBOSE:
            print(maxval)
            print(deplist)
            print(indeplist)
        cX = X[:, deplist]

        condeps = np.empty((len(indeplist), 1))
        maxval = -100
        mostdepL = None
        for l in indeplist:
            cZ = X[:, l]
            tmp = codec3(cZ, Y, cX, device)

            if tmp > maxval:
                maxval = tmp
                mostdepL = l

        # pick randomly (the last one) if all -inf
        if maxval <= -100:
            mostdepL = l

        if maxval <= 0.0 and earlyStop:
            break

        depset.add(mostdepL)
        indepset.remove(mostdepL)

        deplist.append(mostdepL)
        indeplist = list(indepset)

        ordering.append(mostdepL)
        codecVals.append(maxval)

    return ordering, codecVals


def reverseLinearIndexingToLayers(selectedSlices, torchLayers):
    ind_list = []
    for myslice in selectedSlices:
        prevslicecnt = 0
        if isinstance(torchLayers[0], torch.nn.Conv2d):
            nextslicecnt = torchLayers[0].out_channels
        elif isinstance(torchLayers[0], torch.nn.Linear):
            nextslicecnt = torchLayers[0].out_features
        else:
            print(f'cannot reverse process layer: {torchLayers[0]}')
            return NotImplementedError

        for l in range(len(torchLayers)):
            if myslice < nextslicecnt:
                modslice = myslice - prevslicecnt
                ind_list.append([torchLayers[l], modslice])
                break

            prevslicecnt = nextslicecnt

            if isinstance(torchLayers[l+1], torch.nn.Conv2d):
                nextslicecnt += torchLayers[l+1].out_channels
            elif isinstance(torchLayers[l+1], torch.nn.Linear):
                nextslicecnt += torchLayers[l+1].out_features
            else:
                print(f'cannot reverse process layer: {torchLayers[l+1]}')
                return NotImplementedError

    return ind_list


def DisableBatchNorm(model):
    for name, child in (model.named_children()):
        if name.find('BatchNorm') != -1:
            pass
        else:
            child.eval()
            child.track_running_stats = False

    return model


def getVectorizedGrad(gradlist, model, slices_to_update, device, paramlist=None):

    mapDict = {}
    vect_grad = torch.Tensor(0).to(device)
    vect_param = torch.Tensor(0).to(device)

    for layerIdx in range(len(slices_to_update)):

        [layer, sliceID] = slices_to_update[layerIdx]

        for param in layer.named_parameters():

            orig_shape = param[1][sliceID].shape

            if paramlist is not None:
                pparam = paramlist[(str(layer), param[0])]
                vectVersionParam = torch.flatten(pparam[sliceID])
            else:
                vectVersionParam = torch.flatten(param[1][sliceID])

            pgrad = gradlist[(str(layer), param[0])]
            vectVersionGrad = torch.flatten(pgrad[sliceID])

            start_idx = vect_grad.shape[0]
            vect_grad = torch.cat([vect_grad, vectVersionGrad.to(device)], dim=0)
            vect_param = torch.cat([vect_param, vectVersionParam.to(device)], dim=0)
            end_idx = vect_grad.shape[0]

            myKey = (str(layer), param[0], sliceID)
            myVal = [start_idx, end_idx, orig_shape, param[1]]
            mapDict[myKey] = myVal

    return vect_grad, vect_param, mapDict


def getOldPandG(outString, param_bank, grad_bank, epoch, model, slices_to_update, device):
    outString = outString.replace('.pth', '')
    name = outString + '_epoch_' + str(epoch) + "_params.pth"
    paramlist = torch.load(name)
    name = outString + '_epoch_' + str(epoch) + "_grads.pth"
    gradlist = torch.load(name)
    vectG, vectP, _ = getVectorizedGrad(
        grad_bank, model, slices_to_update, device, paramlist=param_bank)

    return vectG, vectP


def getHessian(dw1, dw2=None, approxType='FD', w1=None, w2=None, hessian_device='cpu'):
    original_device = dw1.device
    dw1 = dw1.to(hessian_device)
    dw2 = dw2.to(hessian_device)
    if approxType == 'FD':
        w1 = w1.to(hessian_device)
        w2 = w2.to(hessian_device)

        # hessian = torch.matmul((dw1 - dw2), (dw1 - dw2).transpose())
        grad_diff_outer = torch.einsum('p,q->pq', (dw1-dw2), (dw1-dw2))

        # divide by weight diff:
        pdist = torch.nn.PairwiseDistance(p=1)
        weight_scaling = pdist(w1.view(1, -1), w2.view(1, -1))

        hessian = torch.div(grad_diff_outer, weight_scaling)

    elif approxType == 'Fisher':
        hessian = torch.einsum('p,q->pq', dw1, dw1)

    else:
        print('Unknown Hessian Approximation Type')

    return hessian.to(original_device)


def NewtonScrubStep(weight, grad, hessian, n, l2lambda=0, hessian_device='cpu'):
    original_device = weight.device
    smoothhessian = hessian.to(
        hessian_device) + (l2lambda)*torch.eye(hessian.shape[0]).to(hessian_device)
    newton = torch.linalg.solve(smoothhessian, grad.to(hessian_device))
    newton = newton.to(original_device)
    new_weight = weight + (1/(n-1))*newton
    return new_weight


def NoisyReturn(weights, epsilon=0.1, delta=0.01, m=1, nsamps=50000, M=0.25, L=1.0, lamb=0.01, device='cpu', VERBOSE=False):
    # func params default for cross entropy
    # cross entropy not strongly convex, pick a small number
    gamma = 2*M*(L**2)*(m**2)/((nsamps**2)*(lamb**3))
    sigma = (gamma/epsilon)*np.sqrt(2*np.log(1.25/delta))
    if VERBOSE:
        print('std for noise:', sigma)

    noise = torch.normal(torch.zeros(len(weights)), sigma).to(device)
    return weights + noise


def CR_NaiveNewton(weight, grad, hessian, l2lambda=0, hessian_device='cpu'):
    original_device = weight.device

    smoothhessian = hessian.to(
        hessian_device) + (l2lambda)*torch.eye(hessian.shape[0]).to(hessian_device)
    newton = torch.linalg.solve(smoothhessian, grad.to(
        hessian_device)).to(original_device)

    # removal, towards positive gradient direction
    new_weight = weight + newton

    return new_weight


def updateModelParams(updatedParams, reversalDict, model):
    # Ensure model is in training mode (in case it affects parameter updates)
    model.train()
    for key in reversalDict.keys():
        layername, weightbias, uu = key
        start_idx, end_idx, orig_shape, param = reversalDict[key]
        # slice this update
        vec_w = updatedParams[start_idx:end_idx]
        # reshape
        reshaped_w = vec_w.reshape(orig_shape)
        # We use [:] to copy the data into the existing tensor
        param[uu] = reshaped_w.clone().detach()
        # getattr(model, layername).__getattr__(weightbias)[
        #     uu][:] = reshaped_w.clone().detach()
    # return model


def codec2(Z, Y, device):
    # Y ind Z
    if len(Z.shape) == 1:
        Z = Z.reshape(-1, 1)
    if len(Y.shape) == 2:
        if Y.shape[1] == 1:
            Y = Y.squeeze()
        else:
            print(Y.shape)
            print("Cannot handle multidimensional Y.")

    n, q = Z.shape
    W = Z
    M = OneNN_Torch(W)

    p = torch.argsort(Y)  # ascending
    R = torch.arange(n)
    tmpR = torch.arange(n)
    R = R.to(device)
    tmpR = tmpR.to(device)
    R[p] = tmpR + 1
    RM = R[M]
    minRM = n*torch.minimum(R, RM)

    L = (n+1) - R

    Tn_num = (minRM - L**2).sum()
    Tn_den = torch.dot(L.float(), (n-L).float())

    return Tn_num/Tn_den


def codec3(Z, Y, X, device):
    # Y ind Z given X

    if len(Z.shape) == 1:
        Z = Z.view(-1, 1)
    if len(X.shape) == 1:
        X = X.view(-1, 1)
    if len(Y.shape) == 2:
        if Y.shape[1] == 1:
            Y = Y.squeeze()
        else:
            print(Y.shape)
            print("Cannot handle multidimensional Y.")

    n, px = X.shape

    N = OneNN_Torch(X)

    W = torch.hstack((X, Z))
    M = OneNN_Torch(W)

    p = torch.argsort(Y)  # ascending
    R = torch.arange(n)
    tmpR = torch.arange(n)
    R = R.to(device)
    tmpR = tmpR.to(device)
    R[p] = tmpR + 1

    RM = R[M]
    RN = R[N]
    minRM = torch.minimum(R, RM)
    minRN = torch.minimum(R, RN)

    Tn_num = (minRM - minRN).sum()
    Tn_den = (R - minRN).sum()

    return Tn_num/Tn_den


def OneNN_Torch(X, p=2):
    '''
        Compute pairwise p-norm distance and gets
        elements with closest distance.
    '''

    # number of samples is first dimension
    # feature space size is second dimension
    n, d = X.shape

    pdists = torch.cdist(X, X, p=p,
                         compute_mode='use_mm_for_euclid_dist_if_necessary')

    pdists.fill_diagonal_(float('inf'))
    oneNN = torch.argmin(pdists, dim=1)

    return oneNN
