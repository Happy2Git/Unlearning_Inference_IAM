import copy
import torch
from ..utils import get_model_init


def ascent(net_input, forget_loader, epochs: int = 5, lr: float = 0.01, DEVICE: str = 'cpu'):
    """ Unlearning by Gradient Ascent
    """
    net = copy.deepcopy(net_input)
    net_init = get_model_init(net_input, DEVICE)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net_init.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    net_init.train()
    net.train()

    for _ in range(epochs):
        for inputs, targets in forget_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net_init(inputs)
            loss = criterion(outputs, targets)
            # minus the gradient of sensitive batch
            grads = torch.autograd.grad(loss, net_init.parameters())
            with torch.no_grad():
                for param, grad in zip(net_init.parameters(), grads):
                    if param.grad is not None:
                        param.grad.zero_()
                    param.grad = grad
                # update net with minus gradient
                get_lr = scheduler.get_last_lr()
                for param, grad in zip(net.parameters(), grads):
                    param.data += get_lr[-1]*grad

            optimizer.step()
        scheduler.step()

    net.eval()
    return net
