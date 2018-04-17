#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp

import numpy as np
import torch.optim as optim

from l2o.agent import Agent, _to_cpu
from l2o.args import args
from l2o.env import LR, QuadraticEnvironment, LogisticEnvironment, MlpEnvironment
from l2o.utils import plot_data

import numpy as np
import os.path as osp

from tqdm import trange


def fix_random_seed(seed):
    import random, scipy, numpy as np, torch # pylint: disable=C0410
    random.seed(seed)
    np.random.seed(seed)
    scipy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train(meta_model_path):
    if args.env == 'quadratic':
        env = QuadraticEnvironment(batch_size=args.batch_size, dimension=args.dimension)
    elif args.env == 'logistic':
        env = LogisticEnvironment(batch_size=args.batch_size, dimension=args.dimension)
    elif args.env == 'mlp':
        env = MlpEnvironment(batch_size=args.batch_size, dimension=args.dimension)
    else:
        raise NotImplementedError

    action_size = len(LR)
    state_size = 2 * args.dimension + 1
    agent = Agent(batch_size=args.batch_size, state_size=state_size, action_size=action_size, hidden_size=args.hidden_size).cuda()
    env.cuda()
    agent.cuda()

    optimizer = optim.Adam(agent.all_params, lr=args.lr, eps=1e-5)

    for episode in range(args.n_episodes):
        mean_reward = agent.train_episode(env=env, n_steps=args.n_steps, optim=optimizer)
        current_func_val = env.func_val.cpu().numpy()
        print(f"Episode {episode}, mean reward {mean_reward:.4f}, loss {current_func_val.mean():.4f}")


def test():
    """
    loads meta model from path and compares the optimization using the meta model and with a base model on the same environment
    """
    if args.env == 'quadratic':
        env = QuadraticEnvironment(batch_size=args.batch_size, dimension=args.dimension)
    elif args.env == "logistic":
        raise NotImplementedError
    elif args.env == 'mlp':
        env = MlpEnvironment(batch_size=args.batch_size, dimension=args.dimension)

    env.reset()
    env.cuda()

    base_optimizer = optim.Adam(env.all_params, lr=args.lr, eps=1e-5)
    base_func_vals, base_rewards = [], []
    for step in trange(args.n_steps_test):
        base_optimizer.zero_grad()
        prev_func_vals = env.func_val.cpu()
        next_func_vals = env._eval().cpu()
        base_optimizer.step()
        env.func_val = next_func_vals
        reward = prev_func_vals - next_func_vals
        base_func_vals.append(next_func_vals.cpu().numpy())
        base_rewards.append(reward.cpu().numpy())
    base_func_vals = np.asarray(base_func_vals)
    base_rewards = np.asarray(base_rewards)

    avg_base_func_vals = np.mean(base_func_vals, axis=1)
    avg_base_rewards = np.mean(base_rewards, axis=1)
    print(avg_base_func_vals.tolist())

    data_x = np.arange(args.n_steps_test)
    fig_func_val = plot_data(data_x=data_x, data_y=avg_base_func_vals, label_x='steps', label_y='Func Val', label='adam', fig_no=0)
    fig_rewards = plot_data(data_x=data_x, data_y=avg_base_rewards, label_x='steps', label_y='rewards', label='adam', fig_no=1)

    if args.env == 'quadratic':
        current_x = env.x.data.squeeze(dim=-1).cpu().numpy()
        distance_x = ((current_x - env.x_opt) * (current_x - env.x_opt)).sum(axis=1).mean()
        distance_func_val = (current_func_val - env.f_opt).mean()
        print(f"episode {episode}, mean reward {mean_reward:.4f} func_val {current_func_val.mean():.4f} distance_x {distance_x:.4f} distance_func_val {distance_func_val:.4f} opt_func {env.f_opt.mean():.4f}")
    elif args.env == 'logistic':
        print(f"episode {episode}, mean reward {mean_reward:.4f} func_val {current_func_val.mean():.4f}")
    elif args.env == "mlp":
        print(f"Episode {episode}, mean reward {mean_reward:.4f}, loss {current_func_val.mean():.4f}")
    else:
        raise NotImplementedError

    agent.save(path=meta_model_path)

def test(meta_optimizer_path):
    """
    loads meta model from path and compares the optimization using the meta model and with a base model on the same environment
    """
    if args.env == 'quadratic':
        env = QuadraticEnvironment(batch_size=args.batch_size, dimension=args.dimension)
    if args.env == 'mlp':
        env = MlpEnvironment(batch_size=args.batch_size, dimension=args.dimension)
    elif args.env == 'logistic':
        pass
    env.cuda()
    action_size = len(LR)
    state_size = 2 * args.dimension + 1
    agent = Agent(batch_size=args.batch_size, state_size=state_size, action_size=action_size, hidden_size=args.hidden_size)
    agent.cuda()
    agent.load(meta_optimizer_path)
    env.reset()
    # import ipdb; ipdb.set_trace()
    meta_func_vals, meta_rewards = agent.test_episode(env=env, n_steps=args.n_steps_test, batch_size=args.batch_size) # list of scalars, list of scalars
    meta_func_vals = np.asarray(meta_func_vals)
    meta_rewards = np.asarray(meta_rewards)
    avg_meta_func_vals = np.mean(meta_func_vals, axis=1)
    avg_meta_rewards = np.mean(meta_rewards, axis=1)
    
    env.reset()
    base_optimizer = optim.Adam(env.all_params, lr=args.lr_base, eps=1e-5)
    # base_optimizer = optim.SGD(env.all_params, lr=args.lr_base)
    base_func_vals, base_rewards = [], []
    for step in range(args.n_steps_test):
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
    fig_func_val = plot_data(data_x=data_x, data_y=avg_base_func_vals, label_x='steps', label_y='Func Val', label='adam', fig_no=0)


    fig_rewards = plot_data(data_x=data_x, data_y=avg_meta_rewards, label_x='steps', label_y='rewards', label='meta', fig_no=1)
    fig_rewards = plot_data(data_x=data_x, data_y=avg_base_rewards, label_x='steps', label_y='rewards', label='adam', fig_no=1)

    print(f'saving plot to {osp.join(args.save_dir, "func_val.png")}')
    fig_func_val.savefig(osp.join(args.save_dir, 'func_val.png'))

    print(f'saving plot to {osp.join(args.save_dir, "rewards.png")}')
    fig_rewards.savefig(osp.join(args.save_dir, 'rewards.png'))

def main():
    meta_model_path = f'{args.save_dir}/quadratic_meta_model.pth'
    train(meta_model_path)
    test(meta_model_path)

if __name__ == "__main__":
    main()
