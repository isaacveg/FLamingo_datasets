import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import sys


def get_distinct_colors(num_colors):
    """ Generate distinct colors by using HSV color space and converting to RGB. """
    if num_colors <= 20:
        colors = matplotlib.colormaps['tab20']
        # into list
        colors = [colors(i) for i in range(num_colors)]
    else:
        hues = np.linspace(0, 1, num_colors + 1)[:-1]  # Exclude endpoint to avoid repeating the first color
        colors = [mcolors.hsv_to_rgb([h, 0.5, 0.9]) for h in hues]  # Saturation=0.5, Value=0.9
    return colors

def read_data(file_path):
    filepath = os.path.join(file_path, 'stats.json')
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def get_total_classes(data):
    sets = set()
    for client in data['client_stats']:
        for key in data['client_stats'][client]['train_class_distribution']:
            sets.add(key)
        for key in data['client_stats'][client]['test_class_distribution']:
            sets.add(key)
    return len(sets)

def plot_class_distribution_byclient(file_path):
    """Generate a bar plot of the class distribution on each client.
    Args:
        file_path (str): Path to the directory containing the stats.json file.
    """
    data = read_data(file_path)
    client_stats = data['client_stats']
    # print(client_stats)
    num_clients = len(client_stats)
    num_classes = get_total_classes(data) 
    client_sizes = [client_stats[str(i)]['train_size'] for i in range(1, num_clients+1)]
    
    # Initialize a matrix to store class distributions
    client_distributions = np.zeros((num_clients, num_classes))
    
    # Collect class distribution data
    for i, client_info in enumerate(client_stats.values(), start=0):
        train_dist = client_info['train_class_distribution']
        for cls, count in train_dist.items():
            client_distributions[i, int(cls)] = count
            
    client_totals = client_distributions.sum(axis=1, keepdims=True)
    normalized_distributions = client_distributions / client_totals

    fig, ax = plt.subplots(figsize=(14, 8))
    bottom = np.zeros(num_clients)
    colors = get_distinct_colors(num_classes)

    for i in range(num_classes):
        ax.bar(np.arange(num_clients), normalized_distributions[:, i], bottom=bottom,
               color=colors[i], edgecolor='white', width=0.8, label=f'Class {i}')
        bottom += normalized_distributions[:, i]

    # for idx, size in enumerate(client_sizes):
    #     ax.text(idx, -0.05, str(size), ha='center', va='top', fontsize=8, rotation=45)

    ax.set_title('Proportion of Each Class per Client')
    ax.set_xlabel('Client ID')
    ax.set_ylabel('Proportion of Data')
    ax.set_xticks(np.arange(num_clients))
    ax.set_xticklabels([f'Client {i+1} / {client_sizes[i]}' for i in range(num_clients)], rotation=45, ha="right")
    ax.legend(title='Class ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(file_path,'client_distribution.png'))
    # plt.show()

def plot_class_distribution_byclass(file_path):
    """Generate a bar plot of the class distribution across clients.
    Args:
        file_path (str): Path to the directory containing the stats.json file.
    """
    data = read_data(file_path)
    client_stats = data['client_stats']
    num_clients = len(client_stats)
    num_classes = get_total_classes(data) 

    # Initialize a matrix to store class distributions
    class_distributions = np.zeros((num_classes, num_clients))

    # Collect class distribution data
    for i, client_info in enumerate(client_stats.values(), start=0):
        train_dist = client_info['train_class_distribution']
        for cls, count in train_dist.items():
            class_distributions[int(cls), i] = count

    # Normalize class distributions by the total number of samples per class across all clients
    class_totals = class_distributions.sum(axis=1, keepdims=True)
    normalized_distributions = class_distributions / class_totals

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8))
    # client_colors = plt.cm.get_cmap('tab100', num_clients)
    # client_colors = plt.colormaps['tab100']
    client_colors = get_distinct_colors(num_clients)
    bottom = np.zeros(num_classes)

    # Stack bars for each class across all clients
    for i in range(num_clients):
        ax.bar(np.arange(num_classes), normalized_distributions[:, i], bottom=bottom,
               color=client_colors[i], edgecolor='white', width=0.8, label=f'Client {i+1}')
        bottom += normalized_distributions[:, i]

    ax.set_title('Normalized Class Distribution across Clients')
    ax.set_xlabel('Class ID')
    ax.set_ylabel('Normalized Distribution')
    ax.set_xticks(np.arange(num_classes))
    ax.set_xticklabels(np.arange(num_classes))
    ax.legend(title='Client ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(file_path,'class_distribution.png'))
    # plt.show()

# plot_class_distribution(data)
# plot_class_distribution('.')
# filename = sys.argv[1]
# plot_class_distribution_byclass(filename)
# plot_class_distribution_byclient(filename)
