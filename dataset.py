#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from sklearn.datasets import make_classification

from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

Variable = torch.autograd.Variable


def get_from_nn(model, n_samples, n_features):
    data_x = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    data_x = data_x.astype("float32")
    data_x = Variable(torch.from_numpy(data_x), volatile=True)
    data_y = model(data_x)
    return data_x.data.cpu().numpy(), data_y.data.cpu().numpy()


def get_synthetic(n_samples, n_features, n_classes, n_informative=None, n_clusters_per_class=None, flip_y=None, class_sep=None):
    n_informative = n_informative or n_features
    n_clusters_per_class = n_clusters_per_class or 5
    flip_y = flip_y or 0.1
    class_sep = class_sep or 1.0
    params = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_informative": n_informative,
        "n_redundant": 0,
        "n_repeated": 0,
        "n_classes": n_classes,
        "n_clusters_per_class": n_clusters_per_class,
        "weights": None,
        "flip_y": flip_y,
        "class_sep": class_sep,
    }
    data_x, data_y = make_classification(**params)
    return data_x, data_y


def _extract_from_torch_loader(loader):
    data_x, data_y = [], []
    for sub_data_x, sub_data_y in loader:
        data_x.append(sub_data_x)
        data_y.append(sub_data_y)
    data_x = torch.cat(data_x, dim=0)
    data_y = torch.cat(data_y, dim=0)
    return data_x, data_y


def get_mnist(split):
    assert split in ["train", "test"]
    loader = torch.utils.data.DataLoader(
        MNIST('./data/mnist',
              train=(split == "train"),
              download=True,
              transform=transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize((0.1307,), (0.3081,))
              ])),
        batch_size=1000000,
        shuffle=False,
    )
    return _extract_from_torch_loader(loader)


def get_cifar10(split):
    assert split in ["train", "test"]
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    loader = torch.utils.data.DataLoader(
        CIFAR10('./data/cifar10',
                train=(split == "train"),
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])),
        batch_size=1000000,
        shuffle=False,
    )
    return _extract_from_torch_loader(loader)


def get_cifar100(split):
    assert split in ["train", "test"]
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    loader = torch.utils.data.DataLoader(
        CIFAR100('./data/cifar100',
                 train=(split == "train"),
                 download=True,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=mean, std=std)
                 ])),
        batch_size=1000000,
        shuffle=False,
    )
    return _extract_from_torch_loader(loader)
