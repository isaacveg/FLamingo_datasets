# Generate Shakespeare dataset for Federated Learning settings.
from torchvision import datasets, transforms
import json
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from argparse import ArgumentParser
import os
import numpy as np

from utils.general_utils import save_stats_json
from utils.nlp_utils import word_to_indices, letter_to_index

def process_x(raw_x_batch):
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = np.array(x_batch)
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
    if not os.path.exists(outdir):
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
    else:
        print("Dataset already generated. Do you want to overwrite it? (y/n)")
        ans = input()
        if ans.lower() != 'y':
            exit(0)
        
    # suppose you have already downloaded the dataset 
    # and placed it in the input directory
    raw_train_dir = os.path.join(indir, 'train/')
    raw_test_dir = os.path.join(indir, 'test/')
    train_files = os.listdir(raw_train_dir)
    test_files = os.listdir(raw_test_dir)
    train_file = os.path.join(raw_train_dir, train_files[0])
    test_file = os.path.join(raw_test_dir, test_files[0])
    
    # the files are in json format
    # we use ['user_data'] and ['x'], ['y'] to transform the data
    with open(train_file) as f:
        raw_train_data = json.load(f)
    with open(test_file) as f:
        raw_test_data = json.load(f)

    train_data_ = []
    train_data_len = []
    test_data_ = []

    for k, v in raw_train_data['user_data'].items():
        train_data_.append({'x': process_x(v['x']), 'y': process_y(v['y'])})
        train_data_len.append(len(train_data_[-1]['x']))
    for k, v in raw_test_data['user_data'].items():
        test_data_.append({'x': process_x(v['x']), 'y': process_y(v['y'])})
        
    train_data = []
    test_data = []

    inds = sorted(range(len(train_data_len)), key=lambda k: train_data_len[k])
    for ind in inds:
        train_data.append(train_data_[ind])
        test_data.append(test_data_[ind])
    
    stats = {}
    for idx, train_dict in enumerate(train_data):
        # for NCP tasks, each client should record stats including:
        # train_size, test_size
        stats[idx+1] = {}
        stats[idx+1]['train_size'] = len(train_dict['y'])
        stats[idx+1]['test_size'] = len(test_data[idx]['y'])
        np.savez_compressed(os.path.join(train_dir, f'{idx+1}.npz'), 
                            data=train_dict['x'], 
                            targets=train_dict['y'])
        np.savez_compressed(os.path.join(test_dir, f'{idx+1}.npz'), 
                            data=test_data[idx]['x'], 
                            targets=test_data[idx]['y'])
        
    save_stats_json('shakespeare', stats, outdir, seed=seed, dist='niid', num_clients=len(train_data))
    print(f"Data saved in {outdir}")
    print("Dataset generated successfully!")