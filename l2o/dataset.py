#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import make_classification


def get_synthetic(n_samples, n_features, n_classes, n_informative=None, n_clusters_per_class=None, flip_y=None, class_sep=None):
    data_x = np.zeros((n_samples, 2), dtype="float32")
    data_y = np.random.randint(2, size=n_samples, dtype="int64")
    data_x[range(n_samples), data_y] = 1
    return data_x, data_y
    n_informative = n_informative or n_features
    n_clusters_per_class = n_clusters_per_class or 1
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
    return data_x.astype("float32"), data_y.astype("int64")
