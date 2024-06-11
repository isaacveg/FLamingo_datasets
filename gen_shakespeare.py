# Generate Shakespeare dataset for Federated Learning settings.
import shutil
from torchvision import datasets, transforms
import json
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from argparse import ArgumentParser
import os
import numpy as np

from utils.general_utils import save_stats_json
from utils.nlp_utils import word_to_indices, letter_to_index
from utils.visualize_utils import plot_class_distribution_byclient

def process_x(raw_x_batch):
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = np.array(x_batch)
    # print("x_batch: ", len(x_batch))
    return x_batch

def process_y(raw_y_batch):
    y_batch = [letter_to_index(c) for c in raw_y_batch]
    y_batch = np.array(y_batch)
    return y_batch


if __name__ == '__main__':
    parser = ArgumentParser(description='CIFAR-10 generation for FLamingo')
    # parser.add_argument('--nc', type=int, default=30, help='number of clients')
    parser.add_argument('--seed', type=int, default=2048, help='random seed')
    parser.add_argument('--indir', type=str, default='./utils/leaf_scripts/shakespeare/data/', help='input dataset directory')
    parser.add_argument('--outdir', type=str, default='../datasets/', help='output dataset directory')
    parser.add_argument('-mt', '--minimum_test_samples', type=int, default=0, help='minimum number of test samples per client')
    # usage: bash ./preprocess.sh -s niid --sf 0.2 -k 0 -t sample -tf 0.8
    # non-iid, select 20% percent of clients, keep clients with at least 0 samples, train-test split 80-20
    # Make sure to delete the rem_user_data, sampled_data, test, and train subfolders in the data directory before re-running preprocess.sh
    # usage: python shakespeare.py --seed 2048 --outdir ../datasets/
    args = parser.parse_args()

    # num_clients = args.nc
    seed = args.seed
    indir = args.indir
    outdir = args.outdir    
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    outdir = os.path.join(outdir, f'shakespeare')
    train_dir = os.path.join(outdir, 'train')
    test_dir = os.path.join(outdir, 'test')
    if os.path.exists(outdir):
        print("Dataset already generated. Do you want to overwrite it? (y/n)")
        ans = input()
        if ans.lower() != 'y':
            exit(0)
        else:
            shutil.rmtree(outdir)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

        
    # suppose you have already downloaded the dataset 
    # and placed it in the input directory
    raw_train_dir = os.path.join(indir, 'train/')
    raw_test_dir = os.path.join(indir, 'test/')
    train_files = os.listdir(raw_train_dir)
    test_files = os.listdir(raw_test_dir)
    train_file = os.path.join(raw_train_dir, train_files[0])
    test_file = os.path.join(raw_test_dir, test_files[0])
    print(train_file, test_file)
    
    # the files are in json format
    # we use ['user_data'] and ['x'], ['y'] to transform the data
    with open(train_file) as f:
        raw_train_data = json.load(f)
        # print(raw_train_data['num_samples'])
    with open(test_file) as f:
        raw_test_data = json.load(f)
        # print(raw_test_data['num_samples'])

    train_data_ = []
    train_data_len = []
    test_data_ = []
    test_data_len = []

    for k, v in raw_train_data['user_data'].items():
        train_data_.append({'x': process_x(v['x']), 'y': process_y(v['y'])})
        train_data_len.append(len(train_data_[-1]['x']))
    i=0
    for k, v in raw_test_data['user_data'].items():
        # print(i:=i+1 ,len(v['x']), len(v['y']))
        test_data_.append({'x': process_x(v['x']), 'y': process_y(v['y'])})
        test_data_len.append(len(test_data_[-1]['x']))
        # print(len(test_data_[-1]['x']))
        
    train_data = []
    test_data = []

    inds = sorted(range(len(train_data_len)), key=lambda k: train_data_len[k])
    for ind in inds:
        train_data.append(train_data_[ind])
        test_data.append(test_data_[ind])
        # print(train_data_len[ind], test_data_len[ind], len(train_data[-1]['x']), len(test_data[-1]['x']))
    
    stats = {}
    idx = 0
    for i, train_dict in enumerate(train_data):
        # for NCP tasks, each client should record stats including:
        # train_size, test_size
        if len(test_data[i]['y']) < args.minimum_test_samples:
            continue
        stats[idx+1] = {}
        stats[idx+1]['train_size'] = len(train_dict['y'])
        stats[idx+1]['test_size'] = len(test_data[i]['y'])
        np.savez_compressed(os.path.join(train_dir, f'{idx+1}.npz'), 
                            data=train_dict['x'], 
                            targets=train_dict['y'])
        np.savez_compressed(os.path.join(test_dir, f'{idx+1}.npz'), 
                            data=test_data[i]['x'], 
                            targets=test_data[i]['y'])
        idx += 1
        
    save_stats_json('shakespeare', stats, outdir, seed=seed, dist='niid', num_clients=idx)
    print(f"Data saved in {outdir}")
    # plot_class_distribution_byclient(outdir)
    print("Dataset generated successfully!")