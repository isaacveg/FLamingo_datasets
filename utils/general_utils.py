from matplotlib.rcsetup import all_backends
import numpy as np
import json, os


def split_train_test(indices, train_ratio=0.8):
    """
    Split data list into train and test.
    Args:
        indices: list of indices
        train_size: size of train set
    Returns:
        train_indices: list of indices for train set
        test_indices: list of indices for test set
    """
    tr, te = [], []
    # shuffle
    indices = np.random.permutation(indices)
    tr = indices[:int(len(indices)*train_ratio)]
    te = indices[int(len(indices)*train_ratio):]
    return tr, te

def get_stat(dataset_label, train_idx, test_idx):
    stats = {}
    full_set = np.concatenate([train_idx, test_idx])
    train_set = dataset_label[train_idx]
    test_set = dataset_label[test_idx]
    stats['size'] = len(full_set)
    stats['train_size'] = len(train_idx)
    stats['test_size'] = len(test_idx)
    stats['num_classes'] = len(np.unique(train_set))
    stats['num_train_classes'] = len(np.unique(train_set))
    stats['num_test_classes'] = len(np.unique(test_set))
    stats['train_class_distribution'] = {i: int(sum(train_set == i)) for i in np.unique(train_set).tolist()}
    stats['test_class_distribution'] = {i: int(sum(test_set == i)) for i in np.unique(test_set).tolist()}
    return stats
    
def save_stats_json(dataset_type, stats, outdir, alpha, least_samples, dist, blc, cc, num_clients, seed):
    """Save stats to json file. 
    """
    all_stats = {}
    all_stats['dataset_type'] = dataset_type
    all_stats['alpha'] = alpha
    all_stats['least_samples'] = least_samples
    all_stats['distribution'] = dist
    all_stats['balance'] = blc
    all_stats['class_per_clients'] = cc
    all_stats['num_clients'] = num_clients
    all_stats['seed'] = seed
    all_stats['client_stats'] = stats
    with open(os.path.join(outdir, 'stats.json'), 'w') as f:
        json.dump(all_stats, f, indent=4)

def subset_data(full_target, num_clients, num_classes, dist, blc, cc, alpha=0.1, least_samples=10, partition_matrix=None):
    """
    Subset data for each client based on the distribution type  
    Args:
        full_target: list of labels
        num_clients: number of clients
        num_classes: number of classes
        dist: distribution type
        blc: balance data among clients
        cc: number of classes per client
    Returns:
        map_dict: dictionary with client index as key and list of indices as value
    """
    if dist == 'iid':
        # iid, uniformly assign same distribution of each class of data to clients
        return subset_iid(full_target, num_clients, num_classes, blc)
    elif dist == 'dir':
        # Dirichlet distribution
        return subset_dir(full_target, num_clients, num_classes, blc, alpha)
    elif dist == 'lbs':
        # label-skew distribution, each client has all classes but different number of samples for each class
        return subset_lbs(full_target, num_clients, num_classes, blc, cc)
    elif dist == 'cls':
        # class-skew distribution, each client has different number of classes
        return subset_cls(full_target, num_clients, num_classes, blc, cc)
    elif partition_matrix is not None:
        return subset_matrix(full_target, num_clients, num_classes, partition_matrix)
    
def subset_matrix(full_target, num_clients, num_classes, partition_matrix):
    """
    Partition matrix is a matrix of size num_classes x num_clients.
    Each element PM(i,j) is the ratio of class i samples assigned to client j.
    Every line of the matrix must sum to 1.
    This function will subset the data based on the partition matrix.
    """
    map_dict = {}
    idxs = np.arange(len(full_target))
    class_idxs = [idxs[full_target == i] for i in range(num_classes)]
    # shuffle
    class_idxs = [np.random.permutation(class_idxs[i]) for i in range(num_classes)]
    for i in range(num_clients):
        map_dict[i] = []
        for cls_idx in range(num_classes):
            cls_portion = int(len(class_idxs[cls_idx]) * partition_matrix[cls_idx, i])
            map_dict[i] += class_idxs[cls_idx][:cls_portion].tolist()
            class_idxs[cls_idx] = class_idxs[cls_idx][cls_portion:]
    return map_dict
    
def subset_iid(full_target, num_clients, num_classes, blc, partition_matrix=None):
    map_dict = {}
    idxs = np.arange(len(full_target))
    class_idxs = [idxs[full_target == i] for i in range(num_classes)]
    # shuffle
    class_idxs = [np.random.permutation(class_idxs[i]) for i in range(num_classes)]
    # balanced data samples
    ## each client has 1/num_clients of each class
    if blc:
        for cls_idx in range(num_classes):
            cls_portion = len(class_idxs[cls_idx]) // num_clients
            for i in range(num_clients):
                map_dict[i] = map_dict.get(i, []) + class_idxs[cls_idx][i*cls_portion:(i+1)*cls_portion].tolist()
    # unbalanced data samples
    ## each client has different total samples, but distribution of each class is same
    else:
        # randomly generate portions for portion of 1 class on each client
        class_idxs_size = [len(class_idxs[i]) for i in range(num_classes)]
        minimum_class_size = min(class_idxs_size)
        cls_portion_list = np.random.dirichlet(np.ones(num_clients))
        # if any class on any client has less samples than 1, regenerate the portion
        while min(cls_portion_list)*minimum_class_size < 1:
            cls_portion_list = np.random.dirichlet(np.ones(num_clients))
        for i in range(num_clients):
            for cls_idx in range(num_classes):
                cls_portion = int(cls_portion_list[i] * class_idxs_size[cls_idx])
                map_dict[i] = map_dict.get(i, []) + class_idxs[cls_idx][:cls_portion].tolist()
                class_idxs[cls_idx] = class_idxs[cls_idx][cls_portion:]
    return map_dict

def subset_dir(full_target, num_clients, num_classes, blc, alpha=0.1):
    # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
    min_size = 0
    dataidx_map = {}
    K = num_classes
    N = len(full_target)

    while min_size < num_classes:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = np.where(full_target == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            if blc:
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_clients):
        dataidx_map[j] = idx_batch[j]
    return dataidx_map    
    
        
def subset_lbs(full_target, num_clients, num_classes, main_ratio=0.4):
    """
    label-skew distribution, each client has all classes but different number of samples for each class.
    Balance is not applicable and num_clients must be divisible by num_classes.
    main_ratio: ratio of samples in main class.
    For example: if class 1 is the main class on client 3, then class 1 samples will occupy main_ratio of the samples on client 3.
    Args:
        full_target: list of labels
        num_clients: number of clients
        num_classes: number of classes
        main_ratio: ratio of samples in main class
    Returns:
        map_dict: dictionary with client index as key and list of indices as value
    """
    map_dict = {}
    idxs = np.arange(len(full_target))
    class_idxs = [idxs[full_target == i] for i in range(num_classes)]
    # shuffle
    class_idxs = [np.random.permutation(class_idxs[i]) for i in range(num_classes)]
    if num_clients % num_classes != 0:
        raise ValueError(f"num_clients must be divisible by num_classes, but got {num_clients} and {num_classes}")
    clients_per_class = num_clients // num_classes
    samples_per_client = len(full_target) // num_clients
    for i in range(num_clients):
        map_dict[i] = []
        for cls_idx in range(num_classes):
            if i % num_classes == cls_idx:
                cls_portion = int(samples_per_client * main_ratio)
            else:
                cls_portion = int(samples_per_client * (1-main_ratio) / (num_classes-1))
            map_dict[i] += class_idxs[cls_idx][:cls_portion].tolist()
            class_idxs[cls_idx] = class_idxs[cls_idx][cls_portion:]
    return map_dict
    
       
def subset_cls(full_target, num_clients, num_classes, blc, cc, least_samples=10):
    """ Class-skew distribution, each client has different number of classes
    Args:
        full_target: list of labels
        num_clients: number of clients
        num_classes: number of classes
        blc: balance data among clients
        cc: number of classes per client
        least_samples: minimum number of samples
    Returns:
        map_dict: dictionary with client index as key and list of indices as value
    """
    map_dict = {}
    idxs = np.arange(len(full_target))
    class_idxs = [idxs[full_target == i] for i in range(num_classes)]
    # shuffle
    class_idxs = [np.random.permutation(class_idxs[i]) for i in range(num_classes)]
    class_num_per_client = [cc for _ in range(num_clients)]
    for i in range(num_classes):
        selected_clients = []
        for client in range(num_clients):
            if class_num_per_client[client] > 0:
                selected_clients.append(client)
        # there are total clients*cc copies of total class needed to be distributed by num_classes.
        select_num = int(np.ceil(num_clients*cc/num_classes))
        selected_clients = sorted(selected_clients, key=lambda x: class_num_per_client[x], reverse=True)
        # selected_clients = np.random.permutation(selected_clients)[:select_num]
        selected_clients = selected_clients[:select_num]
        # selected_clients = selected_clients[:int(np.ceil(num_clients*cc/num_classes))]
        # selected_clients = np.random.permutation(selected_clients)[:select_num]
        # selected_clients = np.random.choice(selected_clients, select_num, replace=False)
        print(selected_clients, i)
        num_all_samples = len(class_idxs[i])
        num_selected_clients = len(selected_clients)
        num_per = num_all_samples / num_selected_clients
        if blc:
            num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
        else:
            num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
        num_samples.append(num_all_samples-sum(num_samples))

        idx = 0
        for client, num_sample in zip(selected_clients, num_samples):
            if client not in map_dict.keys():
                map_dict[client] = class_idxs[i][idx:idx+num_sample]
            else:
                map_dict[client] = np.append(map_dict[client], class_idxs[i][idx:idx+num_sample], axis=0)
            idx += num_sample
            class_num_per_client[client] -= 1
        # print(class_num_per_client)
            
    return map_dict