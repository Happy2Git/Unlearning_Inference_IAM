import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import datasets, transforms
import numpy as np
import copy
from collections import defaultdict


def forsaken(net, forget_loader, retain_loader, nonmem_loader, epochs=100, DEVICE: str = 'cpu'):
    net = net.cpu()
    data_list_V = []
    target_list_V = []
    data_list_P = []
    target_list_P = []
    data_list_w = []
    target_list_w = []

    for data, target in forget_loader:
        data_list_V.append(data)
        target_list_V.append(target)

    D_0 = 10

    for count, (data, target) in enumerate(retain_loader):
        if count >= D_0:
            break
        data_list_w.append(data)
        target_list_w.append(target)

    for count, (data, target) in enumerate(nonmem_loader):
        if count >= D_0:
            break
        data_list_P.append(data)
        target_list_P.append(target)
    data_tensor_V = torch.cat(data_list_V, dim=0)
    data_tensor_P = torch.cat(data_list_P, dim=0)
    data_tensor_w = torch.cat(data_list_w, dim=0)
    target_tensor_P = torch.cat(target_list_P, dim=0)

    exists = torch.any(torch.eq(target_tensor_P, 29))
    print(exists)
    target_tensor_w = torch.cat(target_list_w, dim=0)

    output_P = net(data_tensor_P)

    size = output_P.size()
    class_num = size[1]
    non_exit = []
    flag = False
    for i in range(class_num):
        exists = torch.any(torch.eq(target_tensor_P, i))
        if not exists:
            flag = True
            non_exit.append(i)
    class_probs = []
    class_preds = []
    class_prob, class_pred = torch.max(output_P, dim=1)
    class_probs.append(class_prob)
    class_preds.append(class_pred)
    class_probs = torch.cat(class_probs)
    class_preds = torch.cat(class_preds)

    exists = torch.any(torch.eq(class_preds, 29))
    print(exists)
    average_probs_per_class = defaultdict(list)
    for pred, prob in zip(target_tensor_P, class_probs):
        average_probs_per_class[pred.item()].append(prob.item())

    average_probs = {k: np.mean(v) for k, v in average_probs_per_class.items()}

    total_sum = sum(average_probs.values())
    probabilities_P_dict = {k: v/total_sum for k, v in average_probs.items()}
    values_list = list(probabilities_P_dict.values())
    probabilities_P = torch.Tensor(values_list)

    G = Generator()

    optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)

    lambda_panalty = 0.01
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    net_cur = copy.deepcopy(net)
    net_cur.train()
    output_V = net_cur(data_tensor_V)

    probabilities_V = output_V.mean(dim=0)
    best_loss = 100000.0

    for i in range(epochs):
        optimizer.zero_grad()
        parameters = torch.cat([x.view(-1) for x in net_cur.parameters()])
        estimated_gradients = G(parameters, probabilities_V)
        reshaped_grads = restore_shape(
            list(net_cur.parameters()), estimated_gradients)
        net_cur, _ = apply_gradients(net_cur, reshaped_grads)

        output_V = net_cur(data_tensor_V)
        probabilities_V = output_V.mean(dim=0)
        output_w = net_cur(data_tensor_w)
        part_1 = criterion(output_w, target_tensor_w)
        omega = torch.sum(part_1) / float(D_0)
        mask_grad_l1_norm = estimated_gradients.norm(p=1)
        log_V = F.log_softmax(probabilities_V, dim=0)
        if flag and i==0:
            for m in range(len(non_exit)):
                part1 = probabilities_P[:non_exit[m]]
                part2 = probabilities_P[non_exit[m]:]
                value_tensor = torch.tensor([0.0])
                probabilities_P = torch.cat((part1, value_tensor, part2), dim=0)

        kl_divergence = F.kl_div(log_V, probabilities_P)
        loss = kl_divergence + lambda_panalty * omega * mask_grad_l1_norm
        if loss < best_loss:
            loss.backward(retain_graph=True)
            optimizer.step()
            print("loss: ", loss.item())
            best_loss = loss
            net = copy.deepcopy(net_cur)
            net = net.to(DEVICE)
            net.eval()
            return net
        else:
            net_cur, _ = reset_gradients(net_cur, reshaped_grads)



def create_subset(dataset, num_samples_per_class):
    class_counts = {}
    indices = []

    for i in range(len(dataset)):
        label = dataset[i][1]

        if label in class_counts and class_counts[label] < num_samples_per_class:
            class_counts[label] += 1
            indices.append(i)
        elif label not in class_counts:
            class_counts[label] = 1
            indices.append(i)

        if all(count == num_samples_per_class for count in class_counts.values()):
            break

    return Subset(dataset, indices)


def restore_shape(original_params, new_values):
    result_params = []
    start_index = 0
    for param in original_params:
        length = param.numel()
        result_params.append(
            new_values[start_index:start_index+length].view(param.shape))
        # result_params.append(new_values[start_index:start_index+length])
        start_index += length
    return result_params


def apply_gradients(net, gradients):
    index = 0
    learning_rate = 0.01
    for param in net.parameters():
        param_length = param.nelement()
        param_grads = gradients[index]
        param.data = param.data - learning_rate * param_grads
        index += 1
    net_stats = net.state_dict()
    return net, net_stats


def reset_gradients(net, gradients):
    index = 0
    learning_rate = 0.01
    for param in net.parameters():
        param_length = param.nelement()
        param_grads = gradients[index]
        param.data = param.data + learning_rate * param_grads
        index += 1
    net_stats = net.state_dict()
    return net, net_stats


# forsaken
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = torch.nn.Linear(20, 64)
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, parameters, probabilities):
        x = torch.cat((parameters, probabilities), dim=0)
        self.fc1 = torch.nn.Linear(x.size(0), 64)
        x = self.fc1(x)
        x = torch.relu(x)
        self.fc2 = torch.nn.Linear(64, parameters.size(0))
        x = self.fc2(x)
        return x