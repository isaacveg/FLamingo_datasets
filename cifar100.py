# Generate CIFAR-10 dataset for Federated Learning settings.
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR100
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from argparse import ArgumentParser
import os
import numpy as np

from utils.general_utils import subset_data, split_train_test, get_stat, save_stats_json


if __name__ == '__main__':
    parser = ArgumentParser(description='CIFAR-10 generation for FLamingo')
    parser.add_argument('--nc', type=int, default=30, help='number of clients')
    parser.add_argument('--dist', type=str, default='iid', choices=['iid','dir','lbs','cls'], help='distribution of data among clients')
    parser.add_argument('--blc', type=int, default=1, help='balance data among clients')
    parser.add_argument('--seed', type=int, default=2048, help='random seed')
    parser.add_argument('--cc', type=int, default=50, help='number of classes per client')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha for dirichlet distribution')
    parser.add_argument('--least_samples', type=int, default=100, help='minimum number of samples per class')
    parser.add_argument('--indir', type=str, default='../datasets/', help='input dataset directory')
    parser.add_argument('--outdir', type=str, default='../datasets/', help='output dataset directory')
    # usage: python cifar100.py --nc 30 --dist iid --blc 1 --seed 2048 --cc 50 --alpha 0.1 --least_samples 100 --indir ../datasets/ --outdir ../datasets/
    args = parser.parse_args()

    num_clients = args.nc
    dist = args.dist
    blc = args.blc
    cc = args.cc
    seed = args.seed
    alpha = args.alpha
    least_samples = args.least_samples
    indir = args.indir
    outdir = args.outdir
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    outdir = os.path.join(outdir, f'cifar100_nc{num_clients}_dist{dist}_blc{blc}')
    train_dir = os.path.join(outdir, 'train')
    test_dir = os.path.join(outdir, 'test')
    if not os.path.exists(outdir):
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
    else:
        print("Dataset already generated.")
        exit(0)

    # Get Cifar100 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = CIFAR100(
        root=indir, train=True, download=True, transform=transform)
    testset = CIFAR100(
        root=indir, train=False, download=True, transform=transform)
    trainloader = DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # Save full dataset as test/0.npz or train/0.npz
    np.savez_compressed(os.path.join(train_dir, '0.npz'), data=trainset.data, targets=trainset.targets)
    np.savez_compressed(os.path.join(test_dir, '0.npz'), data=testset.data, targets=testset.targets)
    
    # subset data
    map_dict = subset_data(
        dataset_label, num_clients, 100, dist, blc, cc, alpha, least_samples
    )
    # split data
    stats = {}
    for i in range(num_clients):
        train_idx, test_idx = split_train_test(map_dict[i], train_ratio=0.8)
        # get stats
        stats[i+1] = get_stat(dataset_label, train_idx, test_idx)
        # save data
        np.savez_compressed(os.path.join(train_dir, f'{i+1}.npz'), 
                            data=dataset_image[train_idx], 
                            targets=dataset_label[train_idx])
        np.savez_compressed(os.path.join(test_dir, f'{i+1}.npz'),
                            data=dataset_image[test_idx], 
                            targets=dataset_label[test_idx])
        
    save_stats_json('cifar10', stats, outdir, alpha, least_samples, dist, blc, cc, num_clients, seed)
    print(f"Data saved in {outdir}")
    
    print("Dataset generated successfully!")