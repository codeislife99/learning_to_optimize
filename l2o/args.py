#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--dimension', type=int, default=100)
parser.add_argument('--hidden_size', type=int, default=10)
parser.add_argument('--n_episodes', type=int, default=10000)
parser.add_argument('--n_steps', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--env', type=str, choices=['quadratic, logisitic'], default='quadratic')
# parser.add_argument('--gamma', type=float, default=0.99) # hard coded
args = parser.parse_args()
