#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch.optim as optim
from l2o.agent import Agent, _to_cpu
from l2o.args import args
from l2o.env import LR, QuadraticEnvironment, LogisticEnvironment, MlpEnvironment
from l2o.utils import plot_data, create_dir

import numpy as np
import os.path as osp
from tqdm import trange

def test(meta_model_dir):
    """
    loads meta model from path and compares the optimization using the meta model and with a base model on the same environment
    """
    if args.env == 'quadratic':
        env = QuadraticEnvironment(batch_size=args.batch_size, dimension=args.dimension)
    if args.env == 'mlp':
        env = MlpEnvironment(batch_size=args.batch_size, dimension=args.dimension)
    elif args.env == 'logistic':
        env = LogisticEnvironment(batch_size=args.batch_size, dimension=args.dimension)

    action_size = len(LR)
    state_size = 2 * args.dimension + 1
    agent = Agent(batch_size=args.batch_size, state_size=state_size, action_size=action_size, hidden_size=args.hidden_size)
    agent.load(f"{meta_model_dir}/model.pth")    
    agent.cuda()

    env.reset()
    env.cuda()

    # import ipdb; ipdb.set_trace()
    meta_func_vals, meta_rewards = agent.test_episode(env=env, n_steps=args.n_steps_test, batch_size=args.batch_size) # list of scalars, list of scalars
    meta_func_vals = np.asarray(meta_func_vals)
    meta_rewards = np.asarray(meta_rewards)
    avg_meta_func_vals = np.mean(meta_func_vals, axis=1)
    avg_meta_rewards = np.mean(meta_rewards, axis=1)
    env.reset()
    env.cuda()

    if args.optim_base == 'adam':
        base_optimizer = optim.Adam(env.all_params, lr=args.lr_base, eps=1e-5)
    elif args.optim_base == 'sgd':
        base_optimizer = optim.SGD(env.all_params, lr=args.lr_base)

    base_func_vals, base_rewards = [], []
    # import ipdb; ipdb.set_trace()
    for step in trange(args.n_steps_test):
        base_optimizer.zero_grad()
        prev_func_vals = env.func_val
        next_func_vals = env._eval()
        base_optimizer.step()
        env.func_val = next_func_vals
        reward = prev_func_vals - next_func_vals
        base_func_vals.append(_to_cpu(next_func_vals))
        base_rewards.append(_to_cpu(reward))
    base_func_vals = np.asarray(base_func_vals)
    base_rewards = np.asarray(base_rewards)

    avg_base_func_vals = np.mean(base_func_vals, axis=1)
    avg_base_rewards = np.mean(base_rewards, axis=1)

    data_x = np.arange(args.n_steps_test)
    fig_func_val = plot_data(data_x=data_x, data_y=avg_meta_func_vals, label_x='steps', label_y='Func Val', label='meta', fig_no=0)
    fig_func_val = plot_data(data_x=data_x, data_y=avg_base_func_vals, label_x='steps', label_y='Func Val', label=f'{args.optim_base}', fig_no=0)


    fig_rewards = plot_data(data_x=data_x, data_y=avg_meta_rewards, label_x='steps', label_y='rewards', label='meta', fig_no=1)
    fig_rewards = plot_data(data_x=data_x, data_y=avg_base_rewards, label_x='steps', label_y='rewards', label=f'{args.optim_base}', fig_no=1)

    print(f"meta val: {avg_meta_func_vals[-1]:.4f}, base val: {avg_base_func_vals[-1]:.4f}")

    print(f'saving plot to {meta_model_dir}/func_val_{args.optim_base}_lr_{args.lr_base}_s_{args.n_steps_test}.png')
    fig_func_val.savefig(f'{meta_model_dir}/func_val_{args.optim_base}_lr_{args.lr_base}_s_{args.n_steps_test}.png')

    print(f'saving plot to {meta_model_dir}/rewards_{args.optim_base}_lr_{args.lr_base}_s_{args.n_steps_test}.png')
    fig_rewards.savefig(f'{meta_model_dir}/rewards_{args.optim_base}_lr_{args.lr_base}_s_{args.n_steps_test}.png')
    

def main():
    
    meta_model_dir = f'{args.save_dir}/{args.env}/lr_{args.lr}_bs_{args.batch_size}_dim_{args.dimension}_hid_{args.hidden_size}_gamma_{args.gamma}_eps_{args.n_episodes}_steps_{args.n_steps}/'
    assert(osp.exists(f"{meta_model_dir}/model.pth"))
    print(f'meta_model_dir: {meta_model_dir}')
    test(meta_model_dir)

if __name__ == "__main__":
    main()