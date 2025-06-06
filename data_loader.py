import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
import pickle

class SimpleDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data (numpy.ndarray): A numpy array containing your data.
            labels (numpy.ndarray): A numpy array containing your labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.targets = labels
        self.transform = transform

    def __len__(self):
        # Return the number of samples in your dataset
        return len(self.targets)

    def __getitem__(self, idx):
        # Retrieve the sample and label at the given index
        sample = self.data[idx]
        label = self.targets[idx]

        if self.transform:
            # Apply the transform if one is provided
            sample = self.transform(sample)

        return sample, label


def split_dataset(dataname, train_set, held_out, SHADOW_SIZE=20000, unlearn_type='set_random', forget_class=None, forget_size=500, batch_size=128, num_workers=2, shuffle=True, SEED=42, VAL_SIZE=5000):
    # for the unlearning algorithm we'll also need a split of the train set into
    # forget_set and a retain_set
    RNG_init = torch.Generator()
    RNG_init.manual_seed(42)
    print("len of train_set: ", len(train_set))
    SPLIT_LIST = [SHADOW_SIZE, len(train_set) - SHADOW_SIZE]
    shadow_set, cut_train_set = torch.utils.data.random_split(
        train_set, SPLIT_LIST, generator=RNG_init)

    RNG_forget = torch.Generator()
    RNG_forget.manual_seed(SEED)
    original_targets = train_set.targets  # or train_set.dataset.targets if train_set is a DataLoader
    new_train_indices = cut_train_set.indices  # get the indices of the new train set in the original dataset
    if dataname == 'incremental':
        # get incremental indices in cut_train_set and fetch from train_set
        new_train_indices = new_train_indices[:5000]
        cut_train_set = torch.utils.data.Subset(train_set, new_train_indices)

    shuffled_indices = torch.randperm(len(new_train_indices)-100, generator=RNG_forget).tolist()
    shuffled_indices = shuffled_indices + list(range(len(new_train_indices)-100, len(new_train_indices)))
    
    if unlearn_type=='set_random':    
        FORGET_SIZE = forget_size
        RETAIN_SIZE = len(new_train_indices) - FORGET_SIZE
        SPLIT_LIST = [RETAIN_SIZE, FORGET_SIZE]
        forget_indices = [new_train_indices[i] for i in shuffled_indices[-FORGET_SIZE:]]
        retain_indices = [new_train_indices[i] for i in shuffled_indices[:-FORGET_SIZE]]
        forget_set = torch.utils.data.Subset(train_set, forget_indices)
        retain_set = torch.utils.data.Subset(train_set, retain_indices)
        unlearn_flags = torch.zeros(len(new_train_indices))
        unlearn_flags[shuffled_indices[-FORGET_SIZE:]] = 1
        
        retain_loader = torch.utils.data.DataLoader(
            retain_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=RNG_forget
        )
        forget_loader = torch.utils.data.DataLoader(
            forget_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=RNG_forget
        )

    elif unlearn_type=='one_class':

        print(f'new_train_indices: {len(new_train_indices)} {len(cut_train_set)}')
        print(f'original_targets: {len(original_targets)} {len(train_set)}')
        if isinstance(forget_class, list):
            list_class_idx, calmulate_list_class_idx = [], []
            list_retain_indices = []
            list_forget_indices, calmulate_list_indices = [], []
            list_retain_set = []
            list_forget_set = []
            list_unlearn_flags = []
            list_keep_indices = []
            list_retain_loaders = []
            list_forget_loaders = []
            for i in range(len(forget_class)):
                print(f'forget_class: {forget_class[i]}')
                tmp_class_idx = [idx for idx, indice in enumerate(new_train_indices) if original_targets[indice] == forget_class[i]]
                list_class_idx += tmp_class_idx
                calmulate_list_class_idx.extend(list_class_idx)

                list_forget_indices.append([new_train_indices[idx] for idx in tmp_class_idx])
                calmulate_list_indices.extend(list_forget_indices[i])
                # change calmulate_list_indices to a list of indices
                list_retain_indices.append(np.setdiff1d(np.array(new_train_indices), calmulate_list_indices).tolist())

                print(f'forget_indices: {len(list_forget_indices[i])}')
                print(f'retain_indices: {len(list_retain_indices[i])}')
                
                retain_set = torch.utils.data.Subset(train_set, list_retain_indices[i])
                forget_set = torch.utils.data.Subset(train_set, list_forget_indices[i])
                retain_loader = torch.utils.data.DataLoader(
                    retain_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=RNG_forget
                )
                forget_loader = torch.utils.data.DataLoader(
                    forget_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=RNG_forget
                )

                list_retain_set.append(retain_set)
                list_forget_set.append(forget_set)
                list_retain_loaders.append(retain_loader)
                list_forget_loaders.append(forget_loader)

                unlearn_flags = torch.zeros(len(new_train_indices))
                unlearn_flags[[item for item in tmp_class_idx]] = 1
                # current retain_idx
                tmp_retain_idx = np.setdiff1d(np.array(range(len(new_train_indices))), calmulate_list_class_idx).tolist()
                keep_indices = list(set(tmp_retain_idx + tmp_class_idx))
                list_keep_indices.append(keep_indices)
                list_unlearn_flags.append(unlearn_flags[keep_indices])
        else:
            class_idx = [idx for idx, indice in enumerate(new_train_indices) if original_targets[indice] == forget_class]
            forget_indices = [new_train_indices[idx] for idx in class_idx]
            retain_indices = np.setdiff1d(np.array(new_train_indices), forget_indices).tolist()
            forget_set = torch.utils.data.Subset(train_set, forget_indices)
            retain_set = torch.utils.data.Subset(train_set, retain_indices)
            unlearn_flags = torch.zeros(len(new_train_indices))
            unlearn_flags[[item for item in class_idx]] = 1

            retain_loader = torch.utils.data.DataLoader(
                retain_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=RNG_forget
            )
            forget_loader = torch.utils.data.DataLoader(
                forget_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=RNG_forget
            )

    elif unlearn_type=='class_percentage':
        class_idx = [idx for idx, indice in enumerate(new_train_indices) if original_targets[indice] == forget_class]
        class_idx = class_idx[:int(len(class_idx)*forget_size)]
        forget_indices = [new_train_indices[idx] for idx in class_idx]
        print(f'forget_class: {forget_class} len: {len(class_idx)} ratio: {forget_size} ratio_len: {len(forget_indices)}')
        retain_indices = np.setdiff1d(np.array(new_train_indices), forget_indices).tolist()
        forget_set = torch.utils.data.Subset(train_set, forget_indices)
        retain_set = torch.utils.data.Subset(train_set, retain_indices)
        unlearn_flags = torch.zeros(len(new_train_indices))
        unlearn_flags[[item for item in class_idx]] = 1 

        retain_loader = torch.utils.data.DataLoader(
            retain_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=RNG_forget
        )
        forget_loader = torch.utils.data.DataLoader(
            forget_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=RNG_forget
        )

    print(f'init_train_set: {len(cut_train_set)}')
    cut_train_loader = torch.utils.data.DataLoader(
        cut_train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=RNG_forget
    )

    CLASS_VERIFY = False
    if CLASS_VERIFY:
        # print forget_loader label distribution
        print(f'forget_set: {len(forget_set)}')
        label_list = []
        for x,y in forget_loader:
            label_list.append(y)
        
        # print the label distribution of forget_set
        label_list = torch.cat(label_list, dim=0)
        print(f'forget_set label distribution: {label_list.bincount()}')

    test_set, val_set = torch.utils.data.random_split(
        held_out, [0.5, 0.5], generator=RNG_forget)
    test_loader = DataLoader(test_set, batch_size=128,
                            shuffle=False, num_workers=2)
    
    # if the length of val_set is larger than 5000, we only take the first 5000 samples for efficiency
    if len(val_set) > VAL_SIZE:
        val_set = torch.utils.data.Subset(val_set, range(VAL_SIZE))

    val_loader = DataLoader(val_set, batch_size=128,
                            shuffle=False, num_workers=2)
    
    if isinstance(forget_class, list):
        return cut_train_loader, list_retain_loaders, list_forget_loaders, val_loader, test_loader, shadow_set, cut_train_set, list_unlearn_flags, val_set, list_keep_indices
    
    return cut_train_loader, retain_loader, forget_loader, val_loader, test_loader, shadow_set, cut_train_set, unlearn_flags, val_set

def get_data_loaders(dataname, batch_size=128, num_workers=2, unlearn_type='set_random', forget_class=None, forget_size=500, shuffle=True, SEED=42, train_transforms=None, test_transforms=None):
    if dataname == 'cifar10': # training: 50000, test: 10000, default_forget: 500
        if train_transforms is None:
            train_transforms_cifar10 = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        if test_transforms is None:
            test_transforms_cifar10 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        train_set = datasets.CIFAR10(
            root="~/Documents/datasets", train=True, download=True, transform=train_transforms_cifar10
        )
        # we split held out data into test and validation set
        held_out = datasets.CIFAR10(
            root="~/Documents/datasets", train=False, download=True, transform=test_transforms_cifar10
        )
        SHADOW_SIZE = 20000
        # X_train_tensor, Y_train_tensor = 0

    elif dataname == 'cifar100': # training: 50000, test: 10000, default_forget: 500
        if train_transforms is None:
            train_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        if test_transforms is None:
            test_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        train_set = datasets.CIFAR100(
            root="~/Documents/datasets", train=True, download=True, transform=train_transforms
        )
        # we split held out data into test and validation set
        held_out = datasets.CIFAR100(
            root="~/Documents/datasets", train=False, download=True, transform=test_transforms
        )
        SHADOW_SIZE = 20000
        # X_train_tensor, Y_train_tensor = 0
    elif dataname == 'cinic10': # training: 50000, test: 10000, default_forget: 500
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        normalize = transforms.Normalize(mean=cinic_mean, std=cinic_std)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean, std=cinic_std)
        ])

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean, std=cinic_std)
        ])
        train_set = datasets.ImageFolder(root="~/Documents/datasets/cinic-10/train", transform=train_transform)
        validset = datasets.ImageFolder(root="~/Documents/datasets/cinic-10/valid", transform=transform)
        testset = datasets.ImageFolder(root="~/Documents/datasets/cinic-10/test", transform=transform)
        held_out = torch.utils.data.ConcatDataset([validset, testset])
        SHADOW_SIZE = 20000

    elif dataname == 'incremental': # training: 50000, test: 10000, default_forget: 500
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        normalize = transforms.Normalize(mean=cinic_mean, std=cinic_std)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean, std=cinic_std)
        ])

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean, std=cinic_std)
        ])
        train_set = datasets.ImageFolder(root="~/Documents/datasets/cinic-10/train", transform=train_transform)

        if test_transforms is None:
            test_transforms_cifar10 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        # we split held out data into test and validation set
        held_out = datasets.CIFAR10(
            root="~/Documents/datasets", train=False, download=True, transform=test_transforms_cifar10
        )
        SHADOW_SIZE = 20000

    elif dataname == 'texas': # total: 67330x6170, training: 53864x6170, test: 13466x6170, default_forget: 500
        data_set_features = np.load('./texas-data/texas100-features.npy')
        data_set_label = np.load('./texas-data/texas100-labels.npy')
        X = data_set_features.astype(np.float64)
        Y = data_set_label.astype(np.int32)-1
        all_indices = np.arange(len(X))
        np.random.shuffle(all_indices)
        if not os.path.isfile('./texas_shuffle.pkl'):
            all_indices = np.arange(len(X))
            np.random.shuffle(all_indices)
            pickle.dump(all_indices, open('./texas_shuffle.pkl', 'wb'))
        else:
            all_indices = pickle.load(open('./texas_shuffle.pkl', 'rb'))
        from torch.utils.data import TensorDataset
        X_tensor = torch.tensor(X).float()
        Y_tensor = torch.tensor(Y).long()
        total_samples = len(Y_tensor)
        RNG_init = torch.Generator()
        RNG_init.manual_seed(42)
        shuffled_indices = torch.randperm(total_samples, generator=RNG_init)
        train_size = int(total_samples * 0.8)
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:]
        X_train = X_tensor[train_indices]
        Y_train = Y_tensor[train_indices]
        X_val = X_tensor[val_indices]
        Y_val = Y_tensor[val_indices]
        train_set = SimpleDataset(data=X_train, labels=Y_train)
        held_out = SimpleDataset(data=X_val, labels=Y_val)
        SHADOW_SIZE = 20000
        # train_indices = train_set.indices
        # X_train_tensor = X_tensor[train_indices]
        # Y_train_tensor = Y_tensor[train_indices]
    elif dataname == 'location': # total: 5010x446, training: 4008x446, test: 1002x446, default_forget: 50
        data_set_features = np.load('./location-data/location-features.npy')
        data_set_label = np.load('./location-data/location-labels.npy')
        X = data_set_features.astype(np.float64)
        Y = data_set_label.astype(np.int32)
        from torch.utils.data import TensorDataset
        X_tensor = torch.tensor(X).float()
        Y_tensor = torch.tensor(Y).long()
        total_samples = len(Y_tensor)
        RNG_init = torch.Generator()
        RNG_init.manual_seed(42)
        shuffled_indices = torch.randperm(total_samples, generator=RNG_init)
        train_size = int(total_samples * 0.8)
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:]
        X_train = X_tensor[train_indices]
        Y_train = Y_tensor[train_indices]
        X_val = X_tensor[val_indices]
        Y_val = Y_tensor[val_indices]
        train_set = SimpleDataset(data=X_train, labels=Y_train)
        held_out = SimpleDataset(data=X_val, labels=Y_val)
        SHADOW_SIZE = 1000
        # train_indices = train_set.indices
        # X_train_tensor = X_tensor[train_indices]
        # Y_train_tensor = Y_tensor[train_indices]
    elif dataname == 'purchase': # total:197324x600, training: 157859x600, test: 39365x600, default_forget: 1500
        data_set= np.load('./purchase-data/purchase.npy')
        X = data_set[:,1:].astype(np.float64)
        Y = (data_set[:,0]).astype(np.int32)-1
        from torch.utils.data import TensorDataset
        X_tensor = torch.tensor(X).float()
        Y_tensor = torch.tensor(Y).long()
        total_samples = len(Y_tensor)
        RNG_init = torch.Generator()
        RNG_init.manual_seed(42)
        shuffled_indices = torch.randperm(total_samples, generator=RNG_init)
        train_size = int(total_samples * 0.8)
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:]
        X_train = X_tensor[train_indices]
        Y_train = Y_tensor[train_indices]
        X_val = X_tensor[val_indices]
        Y_val = Y_tensor[val_indices]
        train_set = SimpleDataset(data=X_train, labels=Y_train)
        held_out = SimpleDataset(data=X_val, labels=Y_val)
        SHADOW_SIZE = 40000

    print(f'dataname: {dataname}, train_set: {len(train_set)}, held_out: {len(held_out)}')

    return split_dataset(dataname, train_set, held_out, SHADOW_SIZE=SHADOW_SIZE, unlearn_type=unlearn_type, forget_class=forget_class, forget_size=forget_size, SEED=SEED)


def split_dataset_continual(dataname, train_set, held_out, SHADOW_SIZE=20000, unlearn_type='set_random', forget_class=None, forget_size=500, batch_size=128, num_workers=2, shuffle=True, SEED=42, out_of_order=False, VAL_SIZE=5000):
    # for the unlearning algorithm we'll also need a split of the train set into
    # forget_set and a retain_set
    RNG_init = torch.Generator()
    RNG_init.manual_seed(42)
    print("len of train_set: ", len(train_set))
    SPLIT_LIST = [SHADOW_SIZE, len(train_set) - SHADOW_SIZE]
    shadow_set, cut_train_set = torch.utils.data.random_split(
        train_set, SPLIT_LIST, generator=RNG_init)

    RNG_forget = torch.Generator()
    RNG_forget.manual_seed(SEED)
    if SEED >46 and unlearn_type=='set_random':
        if SEED == 47:
            forget_size = 1000
        elif SEED == 48:
            forget_size = 1500
        elif SEED == 49:
            forget_size = 2000
        elif SEED == 50:
            forget_size = 2500
        elif SEED == 51:
            forget_size = 3000


    forget_split_counts = []
    retain_split_counts = []
    if unlearn_type=='set_random':    
        # FORGET_SIZE = forget_size
        # RETAIN_SIZE = len(cut_train_set) - FORGET_SIZE
        # SPLIT_LIST = [RETAIN_SIZE, FORGET_SIZE]
        # retain_set, forget_set = torch.utils.data.random_split(
        #     cut_train_set, SPLIT_LIST, generator=RNG)
        forget_set = []
        retain_set = []

        # 迭代地构建每个 forget_set 和 retain_set
        for size in forget_size:
            retain_size = len(cut_train_set) - size
            split_list = [retain_size, size]

            # 对 cut_train_set 进行随机分割
            current_retain_set, current_forget_set = torch.utils.data.random_split(
                cut_train_set, split_list, generator=RNG_forget)

            forget_split_counts.append(len(current_forget_set))
            retain_split_counts.append(len(current_retain_set))
            print("forget_set length: ", forget_split_counts[-1])
            print("retain_set length: ", retain_split_counts[-1])

            # 将当前的 forget_set 和 retain_set 添加到列表中
            forget_set.append(current_forget_set)
            retain_set.append(current_retain_set)

    elif unlearn_type=='one_class':
        original_targets = train_set.targets  # or train_set.dataset.targets if train_set is a DataLoader
        new_train_indices = cut_train_set.indices  # get the indices of the new train set in the original dataset

        print(f'new_train_indices: {len(new_train_indices)} {len(cut_train_set)}')
        print(f'original_targets: {len(original_targets)} {len(train_set)}')
        forget_set = []
        retain_set = []

        for i in range(len(forget_class)):
            current_forget_classes = forget_class[:i+1]
            forget_idx = [idx for idx in new_train_indices if original_targets[idx] in current_forget_classes]

            retain_idx = np.setdiff1d(np.array(new_train_indices), forget_idx).tolist()

            current_forget_set = torch.utils.data.Subset(train_set, forget_idx)
            current_retain_set = torch.utils.data.Subset(train_set, retain_idx)

            forget_split_counts.append(len(current_forget_set))
            retain_split_counts.append(len(current_retain_set))
            print("forget_set length: ", forget_split_counts[-1])
            print("retain_set length: ", retain_split_counts[-1])

            forget_set.append(current_forget_set)
            retain_set.append(current_retain_set)
        
        print(f"Number of forget sets created: {len(forget_set)}")
        print(f"Number of retain sets created: {len(retain_set)}")

    elif unlearn_type=='class_percentage':
        original_targets = train_set.targets
        new_train_indices = cut_train_set.indices

        forget_set = []
        retain_set = []

        accumulated_forget_idx = []

        for idx, forget_class in enumerate(forget_class):

            current_forget_idx = [index for index in new_train_indices if original_targets[index] == forget_class]
            shuffled_indices = torch.randperm(len(current_forget_idx), generator=RNG_forget)
            selected_indices = shuffled_indices[:int(len(current_forget_idx) * forget_size)]
            current_selected_forget_idx = [current_forget_idx[i.item()] for i in selected_indices]

            accumulated_forget_idx.extend(current_selected_forget_idx)

            current_retain_idx = np.setdiff1d(np.array(new_train_indices), accumulated_forget_idx).tolist()
            current_forget_set = torch.utils.data.Subset(train_set, accumulated_forget_idx)
            current_retain_set = torch.utils.data.Subset(train_set, current_retain_idx)
            forget_set.append(current_forget_set)
            retain_set.append(current_retain_set)
            forget_split_counts.append(len(current_forget_set))
            retain_split_counts.append(len(current_retain_set))
            print("forget_set length: ", forget_split_counts[-1])
            print("retain_set length: ", retain_split_counts[-1])
    
    if out_of_order:
        import itertools
        shuffle = False
        num = len(forget_set)
        print("num: ", num)
        def create_shuffled_sets(set_indices, class_counts, train_set):
            class_start_indices = [0] + class_counts[:-1]
            class_end_indices = class_counts
            class_indices = [set_indices[start:end] for start, end in zip(class_start_indices, class_end_indices)]
            combinations = list(itertools.permutations(class_indices))
            shuffled_sets = []
            for combo in combinations:
                shuffled_indices = sum(combo, [])
                shuffled_set = torch.utils.data.Subset(train_set, shuffled_indices)
                shuffled_sets.append(shuffled_set)
            return shuffled_sets
        last_forget_set_indices = forget_set[-1].indices
        forget_set = create_shuffled_sets(last_forget_set_indices, forget_split_counts, train_set)
        last_element = retain_set[-1]
        # retain_set = [last_element] * len(forget_set)

        indices = last_element.indices
        part_size = len(indices) // num
        parts = [indices[i * part_size:(i + 1) * part_size] for i in range(3)]
        if len(indices) % num != 0:
            parts[-1].extend(indices[num * part_size:])
        permutations = itertools.permutations(parts)
        retain_set = []
        for perm in permutations:
            shuffled_indices = sum(perm, [])
            shuffled_set = torch.utils.data.Subset(last_element.dataset, shuffled_indices)
            retain_set.append(shuffled_set)

    retain_loaders = []
    forget_loaders = []

    for rt_set in retain_set:
        retain_loader = torch.utils.data.DataLoader(
            rt_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
        retain_loaders.append(retain_loader)

    for fg_set in forget_set:
        forget_loader = torch.utils.data.DataLoader(
            fg_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
        forget_loaders.append(forget_loader)

    # 初始化 init_train_sets 和 init_train_loaders 列表
    init_train_sets = []
    init_train_loaders = []

    # 为每个组合创建 ConcatDataset 和 DataLoader
    for rt_set, fg_set in zip(retain_set, forget_set):
        init_train_set = torch.utils.data.ConcatDataset([rt_set, fg_set])
        init_train_loader = torch.utils.data.DataLoader(
            init_train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
        init_train_sets.append(init_train_set)
        init_train_loaders.append(init_train_loader)

    # 现在处理 test_set 和 val_set，这些不需要修改为列表形式
    test_set, val_set = torch.utils.data.random_split(
            held_out, [0.5, 0.5], generator=RNG_forget)
    test_loader = DataLoader(test_set, batch_size=128,
                            shuffle=False, num_workers=2)
    
    if len(val_set) > VAL_SIZE:
        val_set = torch.utils.data.Subset(val_set, range(VAL_SIZE))

    val_loader = DataLoader(val_set, batch_size=128,
                            shuffle=False, num_workers=2)
    # 返回值应相应地被更新

    return init_train_loaders, retain_loaders, forget_loaders, val_loader, test_loader, shadow_set, forget_set, retain_set, val_set

def get_data_loaders_continual(dataname, batch_size=128, num_workers=2, unlearn_type='set_random', forget_class=None, forget_size=500, shuffle=True, SEED=42, train_transforms=None, test_transforms=None, out_of_order=False):
    if dataname == 'cifar10': # training: 50000, test: 10000, default_forget: 500
        if train_transforms is None:
            train_transforms_cifar10 = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        if test_transforms is None:
            test_transforms_cifar10 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        train_set = datasets.CIFAR10(
            root="~/Documents/datasets", train=True, download=True, transform=train_transforms_cifar10
        )
        # we split held out data into test and validation set
        held_out = datasets.CIFAR10(
            root="~/Documents/datasets", train=False, download=True, transform=test_transforms_cifar10
        )
        SHADOW_SIZE = 20000
        # X_train_tensor, Y_train_tensor = 0

    elif dataname == 'cifar100': # training: 50000, test: 10000, default_forget: 500
        if train_transforms is None:
            train_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        if test_transforms is None:
            test_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        train_set = datasets.CIFAR100(
            root="~/Documents/datasets", train=True, download=True, transform=train_transforms
        )
        # we split held out data into test and validation set
        held_out = datasets.CIFAR100(
            root="~/Documents/datasets", train=False, download=True, transform=test_transforms
        )
        SHADOW_SIZE = 20000
        # X_train_tensor, Y_train_tensor = 0

    elif dataname == 'cinic10': # training: 50000, test: 10000, default_forget: 500
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        normalize = transforms.Normalize(mean=cinic_mean, std=cinic_std)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean, std=cinic_std)
        ])

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean, std=cinic_std)
        ])
        train_set = datasets.ImageFolder(root="~/Documents/datasets/cinic-10/train", transform=train_transform)
        validset = datasets.ImageFolder(root="~/Documents/datasets/cinic-10/valid", transform=transform)
        testset = datasets.ImageFolder(root="~/Documents/datasets/cinic-10/test", transform=transform)
        held_out = torch.utils.data.ConcatDataset([validset, testset])
        SHADOW_SIZE = 20000
        
    elif dataname == 'texas': # total: 67330x6170, training: 53864x6170, test: 13466x6170, default_forget: 500
        data_set_features = np.load('./texas-data/texas100-features.npy')
        data_set_label = np.load('./texas-data/texas100-labels.npy')
        X = data_set_features.astype(np.float64)
        Y = data_set_label.astype(np.int32)-1
        all_indices = np.arange(len(X))
        np.random.shuffle(all_indices)
        if not os.path.isfile('./texas_shuffle.pkl'):
            all_indices = np.arange(len(X))
            np.random.shuffle(all_indices)
            pickle.dump(all_indices, open('./texas_shuffle.pkl', 'wb'))
        else:
            all_indices = pickle.load(open('./texas_shuffle.pkl', 'rb'))
        from torch.utils.data import TensorDataset
        X_tensor = torch.tensor(X).float()
        Y_tensor = torch.tensor(Y).long()
        total_samples = len(Y_tensor)
        RNG_init = torch.Generator()
        RNG_init.manual_seed(42)
        shuffled_indices = torch.randperm(total_samples, generator=RNG_init)
        train_size = int(total_samples * 0.8)
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:]
        X_train = X_tensor[train_indices]
        Y_train = Y_tensor[train_indices]
        X_val = X_tensor[val_indices]
        Y_val = Y_tensor[val_indices]
        train_set = SimpleDataset(data=X_train, labels=Y_train)
        held_out = SimpleDataset(data=X_val, labels=Y_val)
        SHADOW_SIZE = 20000
        # train_indices = train_set.indices
        # X_train_tensor = X_tensor[train_indices]
        # Y_train_tensor = Y_tensor[train_indices]
    elif dataname == 'location': # total: 5010x446, training: 4008x446, test: 1002x446, default_forget: 50
        data_set_features = np.load('./location-data/location-features.npy')
        data_set_label = np.load('./location-data/location-labels.npy')
        X = data_set_features.astype(np.float64)
        Y = data_set_label.astype(np.int32)
        from torch.utils.data import TensorDataset
        X_tensor = torch.tensor(X).float()
        Y_tensor = torch.tensor(Y).long()
        total_samples = len(Y_tensor)
        RNG_init = torch.Generator()
        RNG_init.manual_seed(42)
        shuffled_indices = torch.randperm(total_samples, generator=RNG_init)
        train_size = int(total_samples * 0.8)
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:]
        X_train = X_tensor[train_indices]
        Y_train = Y_tensor[train_indices]
        X_val = X_tensor[val_indices]
        Y_val = Y_tensor[val_indices]
        train_set = SimpleDataset(data=X_train, labels=Y_train)
        held_out = SimpleDataset(data=X_val, labels=Y_val)
        SHADOW_SIZE = 1000
        # train_indices = train_set.indices
        # X_train_tensor = X_tensor[train_indices]
        # Y_train_tensor = Y_tensor[train_indices]
    elif dataname == 'purchase': # total:197324x600, training: 157859x600, test: 39365x600, default_forget: 1500
        data_set= np.load('./purchase-data/purchase.npy')
        X = data_set[:,1:].astype(np.float64)
        Y = (data_set[:,0]).astype(np.int32)-1
        from torch.utils.data import TensorDataset
        X_tensor = torch.tensor(X).float()
        Y_tensor = torch.tensor(Y).long()
        total_samples = len(Y_tensor)
        RNG_init = torch.Generator()
        RNG_init.manual_seed(42)
        shuffled_indices = torch.randperm(total_samples, generator=RNG_init)
        train_size = int(total_samples * 0.8)
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:]
        X_train = X_tensor[train_indices]
        Y_train = Y_tensor[train_indices]
        X_val = X_tensor[val_indices]
        Y_val = Y_tensor[val_indices]
        train_set = SimpleDataset(data=X_train, labels=Y_train)
        held_out = SimpleDataset(data=X_val, labels=Y_val)
        SHADOW_SIZE = 40000

    print(f'dataname: {dataname}, train_set: {len(train_set)}, held_out: {len(held_out)}')

    return split_dataset_continual(dataname, train_set, held_out, SHADOW_SIZE=SHADOW_SIZE, unlearn_type=unlearn_type, forget_class=forget_class, forget_size=forget_size, SEED=SEED, out_of_order=out_of_order)
