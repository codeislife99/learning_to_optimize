import numpy as np 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import os
import os.path as osp
import torch


def plot_data(data_x, data_y, label_x, label_y, label, fig_no):
    fig = plt.figure(fig_no, figsize=(10, 10))
    plt.plot(data_x, data_y, label=label)
    plt.legend()
    plt.ylabel(label_y)
    plt.xlabel(label_x)
    return fig

def create_dir(directory):
    if not osp.exists(directory):
        os.makedirs(directory)
        print(f"created directory: {directory}")

def PCA(X, k):
    X_mean = torch.mean(X, dim=0, keepdim=True)
    X = X - X_mean
    U, S, V = torch.svd(torch.t(X))
    return U[:, : k]
