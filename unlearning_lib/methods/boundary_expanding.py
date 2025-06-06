import time
import copy
import torch
import tqdm
from ..models.resnet import ResNetBackbone, LocationBackbone, TexasBackbone, PurchaseBackbone

"""
Reference: Boundary Unlearning: Rapid Forgetting of Deep Networks via Shifting the Decision Boundary
"""


def boundary_expanding(dataset_name, ori_model, forget_loader, class_num, finetune_epochs=10, lr=1e-5, DEVICE: str = 'cpu'):

    num_classes = list(ori_model.named_children()
                       )[-1][1].out_features
    # narrow_model = copy.deepcopy(ori_model).to(DEVICE)
    if dataset_name == 'cifar10' or dataset_name == 'cifar100' or dataset_name == 'cinic10':
        widen_model = ResNetBackbone(ori_model, num_classes=num_classes + 1)
    elif dataset_name == 'location':
        widen_model = LocationBackbone(ori_model, num_classes=num_classes + 1)
    elif dataset_name == 'texas':
        widen_model = TexasBackbone(ori_model, num_classes=num_classes + 1)
    elif dataset_name == 'purchase':
        widen_model = PurchaseBackbone(ori_model, num_classes=num_classes + 1)
    widen_model = widen_model.to(DEVICE)

    classifier = getattr(ori_model, list(ori_model.named_children())[-1][0])
    widen_classifier = getattr(widen_model, list(
        widen_model.named_children())[-1][0])

    for name, params in classifier.named_parameters():
        # print(name, params.data)
        if 'weight' in name:
            widen_classifier.state_dict()['weight'][0:class_num, ] = classifier.state_dict()[
                name][:, ]
        elif 'bias' in name:
            widen_classifier.state_dict()['bias'][0:class_num, ] = classifier.state_dict()[
                name][:, ]

    forget_data_gen = inf_generator(forget_loader)
    batches_per_epoch = len(forget_loader)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        widen_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # centr_optimizer = optimizer_picker(optimization, widen_model.parameters(), lr=0.00001, momentum=0.9)
    # adv_optimizer = optimizer_picker(optimization, adv_model.parameters(), lr=0.001, momentum=0.9)

    for itr in tqdm.tqdm(range(finetune_epochs * batches_per_epoch)):
        x, y = forget_data_gen.__next__()
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        # print the shape of x and y

        widen_logits = widen_model(x)

        # target label
        target_label = torch.ones_like(y, device=DEVICE)
        target_label *= num_classes

        # adv_train
        widen_model.train()
        widen_model.zero_grad()
        optimizer.zero_grad()

        widen_loss = criterion(widen_logits,
                               target_label)

        widen_loss.backward()
        optimizer.step()

    pruned_model = copy.deepcopy(ori_model).to(DEVICE)

    pruned_classifier = getattr(pruned_model, list(
        pruned_model.named_children())[-1][0])
    for name, params in widen_classifier.named_parameters():
        # print(name)
        if 'weight' in name:
            pruned_classifier.state_dict()['weight'][:, ] = widen_classifier.state_dict()[
                name][0:class_num, ]
        elif 'bias' in name:
            pruned_classifier.state_dict()['bias'][:, ] = widen_classifier.state_dict()[
                name][0:class_num, ]

    # pruned_model = torch.nn.Sequential(feature_extrator, pruned_classifier)
    pruned_model = pruned_model.to(DEVICE)

    # mode = 'pruned' if evaluate else ''
    # _, test_acc = eval(model=pruned_model, data_loader=test_loader, mode=mode, print_perform=evaluate, DEVICE=DEVICE,
    #                    name='test set all class')
    # _, forget_acc = eval(model=pruned_model, data_loader=test_forget_loader, mode=mode, print_perform=evaluate,
    #                      DEVICE=DEVICE, name='test set forget class')
    # _, remain_acc = eval(model=pruned_model, data_loader=test_remain_loader, mode=mode, print_perform=evaluate,
    #                      DEVICE=DEVICE, name='test set remain class')
    # _, train_forget_acc = eval(model=pruned_model, data_loader=forget_loader, mode=mode, print_perform=evaluate,
    #                            DEVICE=DEVICE, name='train set forget class')
    # _, train_remain_acc = eval(model=pruned_model, data_loader=train_remain_loader, mode=mode, print_perform=evaluate,
    #                            DEVICE=DEVICE, name='train set remain class')
    # print('test acc:{:.2%}, forget acc:{:.2%}, remain acc:{:.2%}, train forget acc:{:.2%}, train remain acc:{:.2%}'
    #       .format(test_acc, forget_acc, remain_acc, train_forget_acc, train_remain_acc))
    # end = time.time()
    # print('Time Consuming:', end - start, 'secs')

    # torch.save(widen_model, '{}boundary_expand_widen_model.pth'.format(path))
    # torch.save(pruned_model, '{}boundary_expand_pruned_model.pth'.format(path))
    return pruned_model


def inf_generator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()
