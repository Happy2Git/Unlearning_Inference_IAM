import torch
import time
import random
import psutil
import os
import numpy as np
from collections import deque
from unlearning_lib.models.DNN_single import DNNModel_single, Logistic_regression
mini_sigma = 0.001


def deltagrad(model, train_set, DEVICE: str = 'cpu', args=None):
    '''
        Thi method need to first cache the gradient/parameters/learning rate history in a fixed epoch list.
            --add',  action='store_true', help="The flag for incrementally adding training samples, otherwise for incrementally deleting training samples")
            --ratio',  type=float, help="delete rate or add rate")
            --bz',  type=int, help="batch size in SGD")
            --epochs',  type=int, help="number of epochs in SGD")
            --model',  help="name of models to be used")
            --dataset',  help="dataset to be used")
            --wd', type = float, help="l2 regularization")
            --lr', nargs='+', type = float, help="learning rates")
            --lrlen', nargs='+', type = int, help="The epochs to use some learning rate, used for the case with decayed learning rates")
            --GPU', action='store_true', help="whether the experiments run on GPU")
            --GID', type = int, help="Device ID of the GPU")
            --train', action='store_true', help = 'Train phase over the full training datasets')
            --repo', default = gitignore_repo, help = 'repository to store the data and the intermediate results')
            --method', default = baseline_method, help = 'methods to update the models')
            --period', type = int, help = 'period used in deltagrad')
            --init', type = int, help = 'initial epochs used in deltagrad')
            -m', type = int, help = 'history size used in deltagrad')
            --cached_size', type = int, default = 1000, help = 'size of gradients and parameters cached in GPU in deltagrad')  
    '''

    train_X = torch.Tensor(train_set.data)
    train_Y = torch.Tensor(train_set.targets)
    dataset_train = Logistic_regression.MyDataset(train_X, train_Y)

    dataset_train = 0
    full_ids_list = list(range(len(dataset_train.data)))
    delta_data_ids = random_generate_subset_ids2(
        int(len(dataset_train.data)*args.ratio), full_ids_list)  # ratio: delete rate, delta_data_ids: unlearned data ids
    lrs = args.lr
    lrlens = args.lrlen
    lr_lists = get_lr_list(lrs, lrlens)
    dataset_name = args.dataset

    # generate unlearned data ids
    random_ids_all_epochs, _ = generate_random_id_del(
        dataset_train, args.epochs)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr_lists[0], weight_decay=args.wd)
    t1 = time.time()
    # **** #

    model, gradient_list_all_epochs, \
        para_list_all_epochs, \
        learning_rate_all_epochs = model_training_lr_test(random_ids_all_epochs, args.epochs,
                                                          model, dataset_train, len(
                                                              dataset_train), optimizer,
                                                          criterion, args.bz, args.GPU, DEVICE,
                                                          lr_lists, dataset_name)

    # model, gradient_list_all_epochs, \
    #     para_list_all_epochs, \
    #         learning_rate_all_epochs = model_training(train_loader, args.epochs, \
    #                                                             model, optimizer, \
    #                                                                 criterion, args.bz, args.GPU, DEVICE, \
    #                                                                     lr_lists, dataset_name)
    # **** #
    t2 = time.time()

    regularization_coeff = args.wd
    batch_size = 16
    print("training time full::", t2 - t1)

    # Start unlearn
    grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor = cache_grad_para_history(
        para_list_all_epochs, gradient_list_all_epochs, args.cached_size, args.GPU, DEVICE)
    exp_para_list = None
    exp_grad_list = None
    period = args.period
    init_epochs = args.init

    cached_size = args.cached_size
    dim = [len(dataset_train), len(dataset_train[0][0])]
    t1 = time.time()
    # **** #
    model.eval()
    updated_model = model_update_deltagrad(args.epochs, period, 1, init_epochs, dataset_train, model, grad_list_all_epochs_tensor,
                                           para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, delta_data_ids, args.m,
                                           learning_rate_all_epochs, random_ids_all_epochs, batch_size, dim, criterion, optimizer,
                                           regularization_coeff, args.GPU, DEVICE, dataset_name, exp_para_list_all_epochs=exp_para_list,
                                           exp_gradient_list_all_epochs=exp_grad_list)
    # **** #
    t2 = time.time()
    process = psutil.Process(os.getpid())
    print('memory usage::', process.memory_info().rss)
    print('time_deltagrad::', t2 - t1)
    # print("updated_model: ", updated_model)

    return updated_model


def random_generate_subset_ids2(delta_size, all_ids_list):
    num = len(all_ids_list)
    delta_data_ids = set()
    while len(delta_data_ids) < delta_size:
        id = random.randint(0, num-1)
        delta_data_ids.add(all_ids_list[id])
    return torch.tensor(list(delta_data_ids))


def get_lr_list(lrs, lens):
    learning_rates = []
    for i in range(len(lrs)):
        learning_rates.extend([lrs[i]]*lens[i])
    return learning_rates


def get_sorted_random_ids(random_ids_multi_epochs):
    sorted_ids_multi_epochs = []
    for i in range(len(random_ids_multi_epochs)):
        sorted_ids_multi_epochs.append(
            random_ids_multi_epochs[i].numpy().argsort())
    return sorted_ids_multi_epochs


def generate_random_ids_list(dataset_train, epochs):
    random_ids_all_epochs = []
    for i in range(epochs):
        random_ids = torch.randperm(len(dataset_train))
        random_ids_all_epochs.append(random_ids)
    # print("random_ids_all_epochs: ", random_ids_all_epochs[0])
    sorted_random_ids_all_epochs = get_sorted_random_ids(random_ids_all_epochs)
    return random_ids_all_epochs, sorted_random_ids_all_epochs
    # torch.save(random_ids_all_epochs, git_ignore_folder + 'random_ids_multi_epochs')
    # torch.save(sorted_random_ids_all_epochs, git_ignore_folder + 'sorted_ids_multi_epochs')


def generate_random_id_del(dataset_train, epochs):
    return generate_random_ids_list(dataset_train, epochs)


def model_training_lr_test(random_ids_multi_epochs, epoch, net, dataset_train, data_train_size, optimizer, criterion,
                           batch_size, is_GPU, device, lrs, dataset_name):
    #     global cur_batch_win
    net.train()
    gradient_list_all_epochs = []
    para_list_all_epochs = []
    learning_rate_all_epochs = []
    t1 = time.time()
    net.to(device)
    for j in range(epoch):
        random_ids = random_ids_multi_epochs[j]
        learning_rate = lrs[j]
        update_learning_rate(optimizer, learning_rate)
        i = 0
        for k in range(0, data_train_size, batch_size):
            end_id = k + batch_size
            if end_id > data_train_size:
                end_id = data_train_size
            curr_rand_ids = random_ids[k: end_id]
            if not is_GPU:
                images, labels = dataset_train.data[curr_rand_ids].float(
                ), dataset_train.labels[curr_rand_ids].float()
            else:
                images, labels = dataset_train.data[curr_rand_ids].float().to(
                    device), dataset_train.labels[curr_rand_ids].float().to(device)
            images = torch.autograd.Variable(
                images.type(torch.cuda.FloatTensor))
            labels = torch.autograd.Variable(
                labels.type(torch.cuda.FloatTensor))

            optimizer.zero_grad()

            if dataset_name == "MNIST":
                images = torch.reshape(
                    images, (-1, images.shape[1]*images.shape[2]))
            images = images.permute(0, 3, 1, 2)

            # output = net.forward(images)
            output = net(images)
            output = torch.max(output, 1)
            loss = criterion(output.values, labels.to(torch.float))

            if i % 10 == 0:
                print('Train - Epoch %d, Batch: %d, Loss: %f' %
                      (j, i, loss.detach().cpu().item()))
            i += 1
            loss.backward(retain_graph=False)
            append_gradient_list(gradient_list_all_epochs, None,
                                 para_list_all_epochs, net, None, is_GPU, device)
            optimizer.step()
            learning_rate_all_epochs.append(learning_rate)
            del images, labels
            if i >= 100:
                break
    t2 = time.time()
    print("training_time::", (t2 - t1))
    # print("gradient_list_all_epochs: ", gradient_list_all_epochs[18])
    return net, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs


'''pre-fetch parts of the history parameters and gradients into GPU to save the IO overhead'''


def get_model_para_shape_list(para_list):
    shape_list = []
    full_shape_list = []
    total_shape_size = 0
    for para in list(para_list):
        all_shape_size = 1
        for i in range(len(para.shape)):
            all_shape_size *= para.shape[i]
        total_shape_size += all_shape_size
        shape_list.append(all_shape_size)
        full_shape_list.append(para.shape)
    return full_shape_list, shape_list, total_shape_size


def post_processing_gradien_para_list_all_epochs(para_list_all_epochs, grad_list_all_epochs):
    _, _, total_shape_size = get_model_para_shape_list(para_list_all_epochs[0])
    para_list_all_epoch_tensor = torch.zeros(
        [len(para_list_all_epochs), total_shape_size], dtype=torch.double)
    grad_list_all_epoch_tensor = torch.zeros(
        [len(grad_list_all_epochs), total_shape_size], dtype=torch.double)
    for i in range(len(para_list_all_epochs)):
        para_list_all_epoch_tensor[i] = get_all_vectorized_parameters1(
            para_list_all_epochs[i])
        grad_list_all_epoch_tensor[i] = get_all_vectorized_parameters1(
            grad_list_all_epochs[i])
    return para_list_all_epoch_tensor, grad_list_all_epoch_tensor


def cache_grad_para_history(para_list_all_epochs, gradient_list_all_epochs, cached_size, is_GPU, device):
    para_list_all_epoch_tensor, grad_list_all_epoch_tensor = post_processing_gradien_para_list_all_epochs(
        para_list_all_epochs, gradient_list_all_epochs)
    end_cached_id = cached_size
    if end_cached_id > len(para_list_all_epochs):
        end_cached_id = len(para_list_all_epochs)
    para_list_GPU_tensor = para_list_all_epoch_tensor[0:cached_size]
    grad_list_GPU_tensor = grad_list_all_epoch_tensor[0:cached_size]
    para_list_GPU_tensor = para_list_GPU_tensor.to(device)
    grad_list_GPU_tensor = grad_list_GPU_tensor.to(device)
    return grad_list_all_epoch_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor


def model_update_deltagrad(max_epoch, period, length, init_epochs, dataset_train, model, gradient_list_all_epochs_tensor,
                           para_list_all_epochs_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, delta_ids, m,
                           learning_rate_all_epochs, random_ids_multi_super_iterations, batch_size, dim, criterion, optimizer,
                           regularization_coeff, is_GPU, device, dataset_name,
                           exp_para_list_all_epochs, exp_gradient_list_all_epochs):
    '''function to use deltagrad for incremental updates'''
    print("m: ", m)
    para = list(model.parameters())
    use_standard_way = False
    recorded = 0
    full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(
        model.parameters())
    if not is_GPU:
        vec_para_diff = torch.zeros([total_shape_size, 1], dtype=torch.double)
    else:
        vec_para_diff = torch.zeros(
            [total_shape_size, 1], dtype=torch.double, device=device)
    i = 0
    S_k_list = deque()
    Y_k_list = deque()
    remaining_id_bool_tensor = torch.ones(dataset_train.data.shape[0]).bool()
    # set the delta unlearning data ids to False
    remaining_id_bool_tensor[delta_ids.view(-1)] = False
    old_lr = 0
    cached_id = 0
    batch_id = 1
    i = 0
    for k in range(max_epoch):
        random_ids = random_ids_multi_super_iterations[k]
        id_start = 0
        id_end = 0
        print('k::', k)
        curr_init_epochs = init_epochs
        for j in range(0, dataset_train.data.shape[0], batch_size):
            end_id = j + batch_size
            if end_id > dim[0]:
                end_id = dim[0]
            curr_random_ids = random_ids[j:end_id]
            curr_remaining_bool = remaining_id_bool_tensor[curr_random_ids]
            curr_removed_bool = ~curr_remaining_bool
            batch_delta_X = dataset_train.data[curr_random_ids[curr_removed_bool]]
            batch_delta_Y = dataset_train.labels[curr_random_ids[curr_removed_bool]]
            if dataset_name == "MNIST":
                batch_delta_X = torch.reshape(
                    batch_delta_X, (-1, batch_delta_X.shape[1]*batch_delta_X.shape[2]))
            curr_matched_ids_size = torch.sum(curr_removed_bool).item()
            if curr_matched_ids_size > 0:
                if is_GPU:
                    batch_delta_X = batch_delta_X.to(device)
                    batch_delta_Y = batch_delta_Y.to(device)
            if i == len(learning_rate_all_epochs):
                break
            learning_rate = learning_rate_all_epochs[i]

            if end_id - j - curr_matched_ids_size <= 0:
                i += 1
                continue
            if not learning_rate == old_lr:
                update_learning_rate(optimizer, learning_rate)
            old_lr = learning_rate
            if (i-curr_init_epochs) % period == 0:
                recorded = 0
                use_standard_way = True
            if i < curr_init_epochs or use_standard_way == True:
                #                 t7 = time.time()
                '''explicitly evaluate the gradient'''
                batch_remaining_X = dataset_train.data[curr_random_ids[curr_remaining_bool]]
                if dataset_name == "MNIST":
                    batch_remaining_X = torch.reshape(
                        batch_remaining_X, (-1, batch_remaining_X.shape[1]*batch_remaining_X.shape[2]))
                batch_remaining_Y = dataset_train.labels[curr_random_ids[curr_remaining_bool]]
                if is_GPU:
                    batch_remaining_X = batch_remaining_X.to(device)
                    batch_remaining_Y = batch_remaining_Y.to(device)
                # print("start explicit_iters!")

                if exp_gradient_list_all_epochs is None or exp_para_list_all_epochs is None or len(exp_gradient_list_all_epochs) == 0 or len(exp_para_list_all_epochs) == 0:
                    para, _, _, _ = explicit_iters(batch_delta_X, batch_delta_Y, batch_remaining_X, batch_remaining_Y, curr_matched_ids_size, model, para, k, j, m+1, S_k_list,
                                                   Y_k_list, learning_rate, regularization_coeff, para_list_GPU_tensor, grad_list_GPU_tensor, cached_id, full_shape_list, shape_list, is_GPU, device, criterion, optimizer, None, None)
                else:
                    '''batch_delta_X, batch_delta_Y, batch_remaining_X, batch_remaining_Y, curr_matched_ids_size, model, para, k, p, m, S_k_list, Y_k_list, learning_rate, regularization_coeff, para_list_GPU_tensor, grad_list_GPU_tensor, cached_id, full_shape_list, shape_list, is_GPU, device, exp_para_list, exp_gradient_list'''
                    para, _, _, _ = explicit_iters(batch_delta_X, batch_delta_Y, batch_remaining_X, batch_remaining_Y, curr_matched_ids_size, model, para, k, j, m+1, S_k_list, Y_k_list, learning_rate,
                                                   regularization_coeff, para_list_GPU_tensor, grad_list_GPU_tensor, cached_id, full_shape_list, shape_list, is_GPU, device, criterion, optimizer, exp_para_list_all_epochs[i], exp_gradient_list_all_epochs[i])
                # print("end explicit_iters!")
                use_standard_way = False
            else:
                '''use l-bfgs algorithm to evaluate the gradients'''
                gradient_dual = None
                if curr_matched_ids_size > 0:
                    init_model(model, para)
                    compute_derivative_one_more_step(
                        model, batch_delta_X, batch_delta_Y, criterion, optimizer)
                    # gradient_dual = model.get_all_gradient()
                    gradient_dual = Logistic_regression.get_all_gradient(model)
                with torch.no_grad():
                    vec_para_diff = torch.t((get_all_vectorized_parameters1(
                        para) - para_list_GPU_tensor[cached_id]))
                    if (i-curr_init_epochs)/period >= 1:
                        if (i-curr_init_epochs) % period == 1:
                            zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = prepare_hessian_vec_prod0_3(
                                list(S_k_list)[1:], list(Y_k_list)[1:], i, init_epochs, m, is_GPU, device)
                            mat = np.linalg.inv(mat_prime.cpu().numpy())
                            mat = torch.from_numpy(mat)
                            if is_GPU:
                                mat = mat.to(device)
                        hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms1(
                            zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, vec_para_diff, is_GPU, device)
                    else:
                        '''S_k_list, Y_k_list, v_vec, k, is_GPU, device'''
                        hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_3(
                            list(S_k_list)[1:], list(Y_k_list)[1:], vec_para_diff, m, is_GPU, device)
                    exp_gradient, exp_param = None, None
                    if gradient_dual is not None:
                        is_positive, final_gradient_list = compute_grad_final3(get_all_vectorized_parameters1(para), torch.t(hessian_para_prod), get_all_vectorized_parameters1(
                            gradient_dual), grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, is_GPU, device)
                    else:
                        is_positive, final_gradient_list = compute_grad_final3(get_all_vectorized_parameters1(para), torch.t(
                            hessian_para_prod), None, grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, is_GPU, device)
                    if exp_gradient_list_all_epochs is not None and len(exp_gradient_list_all_epochs) > 0:
                        print('gradient diff::', torch.norm(get_all_vectorized_parameters1(
                            exp_gradient_list_all_epochs[i]) + regularization_coeff*get_all_vectorized_parameters1(para) - final_gradient_list))
                        print('para diff::', torch.norm(get_all_vectorized_parameters1(
                            exp_para_list_all_epochs[i]) - get_all_vectorized_parameters1(para)))
                        print('para change::', torch.norm(get_all_vectorized_parameters1(
                            exp_para_list_all_epochs[i]) - para_list_GPU_tensor[cached_id]))
                    vec_para = update_para_final2(
                        para, final_gradient_list, learning_rate, regularization_coeff, exp_gradient, exp_param)
                    para = get_devectorized_parameters(
                        vec_para, full_shape_list, shape_list)
            i = i + 1
            cached_id += 1
            if cached_id % cached_size == 0:
                GPU_tensor_end_id = (batch_id + 1)*cached_size
                if GPU_tensor_end_id > para_list_all_epochs_tensor.shape[0]:
                    GPU_tensor_end_id = para_list_all_epochs_tensor.shape[0]
                para_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(
                    para_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                grad_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(
                    gradient_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                batch_id += 1
                cached_id = 0
            id_start = id_end
    init_model(model, para)
    return model


def compute_grad_final3(para, hessian_para_prod, gradient_dual, grad_list_tensor, para_list_tensor, size1, size2, alpha, beta, is_GPU, device):
    gradients = None
    if gradient_dual is not None:
        hessian_para_prod += grad_list_tensor
        hessian_para_prod += beta*para_list_tensor
        gradients = hessian_para_prod*size1
        gradients -= (gradient_dual + beta*para)*size2
        gradients /= (size1 - size2)
    else:
        hessian_para_prod += (grad_list_tensor + beta*para_list_tensor)
        gradients = hessian_para_prod
    delta_para = para - para_list_tensor
    delta_grad = hessian_para_prod - (grad_list_tensor + beta*para_list_tensor)
    tmp_res = 0
    if torch.norm(delta_para) > torch.norm(delta_grad):
        return True, gradients
    else:
        return False, gradients


def model_update_standard_lib(num_epochs, dataset_train, model, random_ids_multi_epochs, delta_data_ids, batch_size, learning_rate_all_epochs, criterion, optimizer, is_GPU, device, dataset_name, record_params=False):
    count = 0
    elapse_time = 0
    overhead = 0
    overhead3 = 0
    t1 = time.time()
    exp_gradient_list_all_epochs = []
    exp_para_list_all_epochs = []
    old_lr = -1
    random_ids_list_all_epochs = []
    remaining_tensor_bool = torch.ones(dataset_train.data.shape[0]).bool()
    remaining_tensor_bool[delta_data_ids.view(-1)] = False
    for k in range(num_epochs):
        print("epoch::", k)
        random_ids = random_ids_multi_epochs[k]
        for j in range(0, dataset_train.data.shape[0], batch_size):
            end_id = j + batch_size
            if end_id >= dataset_train.data.shape[0]:
                end_id = dataset_train.data.shape[0]
            curr_random_ids = random_ids[j:end_id]
            curr_remaining_tensor = remaining_tensor_bool[curr_random_ids]
            if k == 0 and j == 0:
                print(curr_random_ids[0:50])
            curr_matched_ids_size = torch.sum(curr_remaining_tensor).item()
            if curr_matched_ids_size <= 0:
                count += 1
                continue
            if not is_GPU:
                batch_X = dataset_train.data[curr_random_ids[curr_remaining_tensor]]
                batch_Y = dataset_train.labels[curr_random_ids[curr_remaining_tensor]]
            else:
                batch_X = dataset_train.data[curr_random_ids[curr_remaining_tensor]].to(
                    device)
                batch_Y = dataset_train.labels[curr_random_ids[curr_remaining_tensor]].to(
                    device)
            learning_rate = learning_rate_all_epochs[count]
            if not learning_rate == old_lr:
                update_learning_rate(optimizer, learning_rate)
            old_lr = learning_rate
            optimizer.zero_grad()
            if dataset_name == "MNIST":
                batch_X = torch.reshape(
                    batch_X, (-1, batch_X.shape[1]*batch_X.shape[2]))
            output = model(batch_X)
            loss = criterion(output, batch_Y)
            loss.backward()
            if record_params:
                append_gradient_list(exp_gradient_list_all_epochs, None,
                                     exp_para_list_all_epochs, model, batch_X, is_GPU, device)
            optimizer.step()
            count += 1
    t2 = time.time()
    elapse_time += (t2 - t1)
    print("training time is", elapse_time)
    print("overhead::", overhead)
    print("overhead3::", overhead3)
    return model, count, exp_para_list_all_epochs, exp_gradient_list_all_epochs, random_ids_list_all_epochs


def update_learning_rate(optim, learning_rate):
    for g in optim.param_groups:
        g['lr'] = learning_rate


def append_gradient_list(gradient_list_all_epochs, output_list_all_epochs, para_list_all_epochs, model, X, is_GPU, device):
    gradient_list = []
    para_list = []
    for param in model.parameters():
        # if not is_GPU:
        # gradient_list.append(param.grad.clone())
        # para_list.append(param.data.clone())
        # else:
        gradient_list.append(param.grad.cpu().clone())
        para_list.append(param.data.cpu().clone())
    if output_list_all_epochs is not None:

        output_list, _ = model.get_output_each_layer(X)
        output_list_all_epochs.append(output_list)
    gradient_list_all_epochs.append(gradient_list)
    para_list_all_epochs.append(para_list)


def get_all_vectorized_parameters1(para_list):
    res_list = []
    i = 0
    for param in para_list:
        res_list.append(param.data.view(-1))
        i += 1
    return torch.cat(res_list, 0).view(1, -1)


def clear_gradients(para_list):
    for param in para_list:
        param.grad.zero_()


def explicit_iters(batch_delta_X, batch_delta_Y, batch_remaining_X, batch_remaining_Y,
                   curr_matched_ids_size, model, para, k, p, m, S_k_list, Y_k_list, learning_rate,
                   regularization_coeff, para_list_GPU_tensor, grad_list_GPU_tensor, cached_id, full_shape_list,
                   shape_list, is_GPU, device, criterion, optimizer, exp_para_list, exp_gradient_list):
    if exp_para_list is not None:
        print('para diff::', torch.norm(get_all_vectorized_parameters1(
            exp_para_list) - get_all_vectorized_parameters1(para)))
    init_model(model, para)
    compute_derivative_one_more_step(
        model, batch_remaining_X, batch_remaining_Y, criterion, optimizer)
    # expect_gradients = get_all_vectorized_parameters1(model.get_all_gradient())
    expect_gradients = get_all_vectorized_parameters1(
        Logistic_regression.get_all_gradient(model))
    if exp_gradient_list is not None:
        print('gradient diff::', torch.norm(expect_gradients -
              get_all_vectorized_parameters1(exp_gradient_list)))
    gradient_remaining = 0
    if curr_matched_ids_size > 0:
        clear_gradients(model.parameters())
        compute_derivative_one_more_step(
            model, batch_delta_X, batch_delta_Y, criterion, optimizer)
        # gradient_remaining = get_all_vectorized_parameters1(model.get_all_gradient())
        gradient_remaining = get_all_vectorized_parameters1(
            Logistic_regression.get_all_gradient(model))
    with torch.no_grad():
        curr_para = get_all_vectorized_parameters1(para)
        if k > 0 or (p > 0 and k == 0):
            prev_para = para_list_GPU_tensor[cached_id].to(device)
            # print("curr_para: ",curr_para)
            # print("prev_para: ", prev_para)
            curr_s_list = (curr_para - prev_para)  # + 1e-16
        gradient_full = (expect_gradients*batch_remaining_X.shape[0] + gradient_remaining*curr_matched_ids_size)/(
            batch_remaining_X.shape[0] + curr_matched_ids_size)
        hessian_para_prod = None
        theta_k = 1
        if k > 0 or (p > 0 and k == 0):
            # print(gradient_full)
            # print(grad_list_GPU_tensor[cached_id])
            # print(regularization_coeff)
            # print(curr_s_list)
            curr_y_k = gradient_full - \
                grad_list_GPU_tensor[cached_id] + \
                regularization_coeff*curr_s_list  # + 1e-16
            if len(Y_k_list) >= m:
                curr_len = m
                if len(S_k_list) < m:
                    curr_len = len(S_k_list)
                curr_y_k_bar = curr_y_k
            else:
                sigma_k = torch.mm(curr_y_k, torch.t(
                    curr_s_list))/(torch.mm(curr_s_list, torch.t(curr_s_list)))

                if sigma_k < mini_sigma:
                    sigma_k = mini_sigma
                curr_y_k_bar = curr_y_k
            Y_k_list.append(curr_y_k_bar)
            if len(Y_k_list) > m:
                removed_y_k = Y_k_list.popleft()
                del removed_y_k
            S_k_list.append(curr_s_list)
            if len(S_k_list) > m:
                removed_s_k = S_k_list.popleft()
                del removed_s_k
        para = get_devectorized_parameters((1-learning_rate*regularization_coeff)
                                           * curr_para - learning_rate*expect_gradients, full_shape_list, shape_list)
        del gradient_full
        del gradient_remaining
        del expect_gradients
        del batch_remaining_X
        del batch_remaining_Y
        if curr_matched_ids_size > 0:
            del batch_delta_X
            del batch_delta_Y
        if k > 0 or (p > 0 and k == 0):
            del prev_para
            del curr_para
    return para, cached_id, hessian_para_prod, theta_k


def init_model(model, para_list):
    i = 0
    for m in model.parameters():
        m.data.copy_(para_list[i])
        if m.grad is not None:
            m.grad.zero_()
        m.requires_grad = True
        i += 1


def compute_derivative_one_more_step(model, batch_X, batch_Y, criterion, optimizer):
    optimizer.zero_grad()
    batch_X = batch_X.permute(0, 3, 1, 2)
    batch_X = torch.autograd.Variable(batch_X.type(torch.cuda.FloatTensor))
    batch_Y = torch.autograd.Variable(batch_Y.type(torch.cuda.FloatTensor))
    output = model(batch_X)
    output = torch.max(output, 1)
    loss = criterion(output.values, batch_Y.to(torch.float))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    return loss


def compute_approx_hessian_vector_prod_with_prepared_terms1(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, v_vec, is_GPU, device):
    # if is_GPU:
    p_mat = torch.zeros([zero_mat_dim*2, 1], dtype=torch.double, device=device)
    # else:
    #     p_mat = torch.zeros([zero_mat_dim*2, 1], dtype = torch.double)
    torch.mm(curr_Y_k, v_vec, out=p_mat[0:zero_mat_dim])
    torch.mm(curr_S_k, v_vec*sigma_k, out=p_mat[zero_mat_dim:zero_mat_dim*2])
    p_mat = torch.mm(mat.to(device), p_mat)
    approx_prod = sigma_k*v_vec
    approx_prod -= (torch.mm(torch.t(curr_Y_k), p_mat[0:zero_mat_dim]) + torch.mm(
        sigma_k*torch.t(curr_S_k), p_mat[zero_mat_dim:zero_mat_dim*2]))
    return approx_prod


def prepare_hessian_vec_prod0_3(S_k_list, Y_k_list, i, m, k, is_GPU, device):
    zero_mat_dim = k  # ids.shape[0]
    curr_S_k = torch.cat(list(S_k_list), dim=0)
    curr_Y_k = torch.cat(list(Y_k_list), dim=0)
    S_k_time_Y_k = torch.mm(curr_S_k, torch.t(curr_Y_k))
    S_k_time_S_k = torch.mm(curr_S_k, torch.t(curr_S_k))

    # if is_GPU:
    R_k = np.triu(S_k_time_Y_k.to('cpu').numpy())
    L_k = S_k_time_Y_k - (torch.from_numpy(R_k)).to(device)
    # else:
    #     R_k = np.triu(S_k_time_Y_k.numpy())
    #     L_k = S_k_time_Y_k - torch.from_numpy(R_k)
    D_k_diag = torch.diag(S_k_time_Y_k)
    sigma_k = torch.mm(Y_k_list[-1], torch.t(S_k_list[-1])) / \
        (torch.mm(S_k_list[-1], torch.t(S_k_list[-1])))

    if sigma_k < mini_sigma:
        sigma_k = mini_sigma
    upper_mat = torch.cat([-torch.diag(D_k_diag), torch.t(L_k)], dim=1)
    lower_mat = torch.cat([L_k, sigma_k*S_k_time_S_k], dim=1)
    mat = torch.cat([upper_mat, lower_mat], dim=0)
    return zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat


def cal_approx_hessian_vec_prod0_3(S_k_list, Y_k_list, v_vec, k, is_GPU, device):
    zero_mat_dim = k  # ids.shape[0]
    curr_S_k = torch.t(torch.cat(list(S_k_list), dim=0))
    curr_Y_k = torch.t(torch.cat(list(Y_k_list), dim=0))
    S_k_time_Y_k = torch.mm(torch.t(curr_S_k), curr_Y_k)
    S_k_time_S_k = torch.mm(torch.t(curr_S_k), curr_S_k)

    R_k = np.triu(S_k_time_Y_k.to('cpu').numpy())
    L_k = S_k_time_Y_k - torch.from_numpy(R_k).to(device)
    # else:
    #     R_k = np.triu(S_k_time_Y_k.numpy())
    #     L_k = S_k_time_Y_k - torch.from_numpy(R_k)
    D_k_diag = torch.diag(S_k_time_Y_k)
    sigma_k = torch.mm(Y_k_list[-1], torch.t(S_k_list[-1])) / \
        (torch.mm(S_k_list[-1], torch.t(S_k_list[-1])))
    if sigma_k < mini_sigma:
        sigma_k = mini_sigma

    if is_GPU:
        p_mat = torch.zeros([zero_mat_dim*2, 1],
                            dtype=torch.double, device=device)
    else:
        p_mat = torch.zeros([zero_mat_dim*2, 1], dtype=torch.double)

    tmp = torch.mm(torch.t(curr_Y_k), v_vec)
    p_mat[0:zero_mat_dim] = tmp
    p_mat[zero_mat_dim:zero_mat_dim *
          2] = torch.mm(torch.t(curr_S_k), v_vec)*sigma_k
    # print("torch.diag(D_k_diag): ", torch.diag(D_k_diag))
    # print("torch.t(L_k): ", torch.t(L_k))
    upper_mat = torch.cat([-torch.diag(D_k_diag), torch.t(L_k)], dim=1)
    lower_mat = torch.cat([L_k, sigma_k*S_k_time_S_k], dim=1)
    # print("upper_mat: ", upper_mat)
    # print("lower_mat: ", lower_mat)
    mat = torch.cat([upper_mat, lower_mat], dim=0)
    # print("mat: ", mat)
    mat = np.linalg.inv(mat.cpu().numpy())
    inv_mat = torch.from_numpy(mat)
    if is_GPU:
        inv_mat = inv_mat.to(device)

    p_mat = torch.mm(inv_mat, p_mat).to(device)  # .to(device)

    approx_prod = sigma_k*v_vec - \
        torch.mm(torch.cat([curr_Y_k, sigma_k*curr_S_k], dim=1), p_mat)
    return approx_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat


def compute_model_para_diff(model1_para_list, model2_para_list):
    diff = 0
    norm1 = 0
    norm2 = 0
    all_dot = 0
    for i in range(len(model1_para_list)):
        param1 = model1_para_list[i].to('cpu')
        param2 = model2_para_list[i].to('cpu')
        curr_diff = torch.norm(param1 - param2, p='fro')
        norm1 += torch.pow(torch.norm(param1, p='fro'), 2)
        norm2 += torch.pow(torch.norm(param2, p='fro'), 2)
        all_dot += torch.sum(param1*param2)
        diff += curr_diff*curr_diff
    # print('model difference (l2 norm):', torch.sqrt(diff))


def update_para_final2(para, gradient_list, alpha, beta, exp_gradient, exp_para):
    exp_grad_list = []
    vec_para = get_all_vectorized_parameters1(para)
    vec_para -= alpha*gradient_list
    if exp_gradient is not None:
        print("grad_diff::")
        compute_model_para_diff(gradient_list, exp_grad_list)
        print("here!!")
    return vec_para


def get_devectorized_parameters(params, full_shape_list, shape_list):
    params = params.view(-1)
    para_list = []
    pos = 0
    for i in range(len(full_shape_list)):
        param = 0
        if len(full_shape_list[i]) >= 2:
            curr_shape_list = list(full_shape_list[i])
            param = params[pos: pos+shape_list[i]].view(curr_shape_list)
        else:
            param = params[pos: pos+shape_list[i]].view(full_shape_list[i])
        para_list.append(param)
        pos += shape_list[i]
    return para_list


def model_training(train_loader, epoch, net, optimizer, criterion, batch_size, is_GPU, device, lrs, dataset_name):
    net.train()
    net.to(device)
    gradient_list_all_epochs = []
    para_list_all_epochs = []
    learning_rate_all_epochs = []
    t1 = time.time()
    # 定义损失函数和优化器
    # 训练模型
    for j in range(epoch):
        learning_rate = lrs[j]
        update_learning_rate(optimizer, learning_rate)
        i = 0

        running_loss = 0
        for images, labels in train_loader:
            # Flatten MNIST images into a 784 long vector
            # images = images.view(images.shape[0], -1)
            images = torch.autograd.Variable(
                images.type(torch.cuda.FloatTensor)).to(device)
            labels = torch.autograd.Variable(
                labels.type(torch.cuda.FloatTensor)).to(device)

            # Training pass
            optimizer.zero_grad()

            output = net(images)
            # print("output[0]: ", output[0])
            output = torch.max(output, 1)
            loss = criterion(output.values, labels.to(torch.float))

            # This is where the model learns by backpropagating
            loss.backward()
            append_gradient_list(gradient_list_all_epochs, None,
                                 para_list_all_epochs, net, None, is_GPU, device)
            # And optimizes its weights here
            optimizer.step()
            learning_rate_all_epochs.append(learning_rate)
            running_loss += loss.item()
            i += 1
        else:
            print(f"Training loss: {running_loss/len(train_loader)}")
    return net, gradient_list_all_epochs, para_list_all_epochs, learning_rate_all_epochs
