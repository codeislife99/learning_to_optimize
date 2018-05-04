#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dimension', type=int, default=193)
parser.add_argument('--hidden_size', type=int, default=10)
parser.add_argument('--n_episodes', type=int, default=10000)
parser.add_argument('--n_steps', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--save_dir', type=str, default='logs', help='folder to save models and plots')
parser.add_argument('--env', type=str, choices=['quadratic', 'logistic', 'mlp'], default='mlp')

parser.add_argument('--lr_base', type=float, default=1e-1)
parser.add_argument('--n_steps_test', type=int, default=100, help='number of steps to take in testing')
parser.add_argument('--optim_base', type=str, choices=['adam', 'sgd'], default='adam', help='base optimizer')
parser.add_argument('--load_model_from', type=str, default="")

# for PCA
parser.add_argument('--pca', action="store_true")
parser.add_argument('--pca_init_lr', type=float, default=0.1)
parser.add_argument('--pca_init_steps', type=int, default=100)
parser.add_argument('--pca_dim', type=int, default=50)

# parser.add_argument('--gamma', type=float, default=0.99) # hard coded
args = parser.parse_args()
