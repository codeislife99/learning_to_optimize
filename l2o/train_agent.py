#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.optim as optim

from l2o.agent import Agent, _to_cpu
from l2o.args import args
from l2o.env import LR, QuadraticEnvironment

import numpy as np


def fix_random_seed(seed):
    import random, scipy, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    scipy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)





def train(meta_model_path):
    if args.env == 'quadratic':
        env = QuadraticEnvironment(batch_size=args.batch_size, dimension=args.dimension)
    elif args.env == 'logistic':
        pass

    action_size = len(LR)
    state_size = 2 * args.dimension + 1
    agent = Agent(batch_size=args.batch_size, state_size=state_size, action_size=action_size, hidden_size=args.hidden_size).cuda()
    env.cuda()
    agent.cuda()

    optimizer = optim.Adam(agent.all_params, lr=args.lr, eps=1e-5)

    for episode in range(args.n_episodes):
        mean_reward = agent.train_episode(env=env, n_steps=args.n_steps, optim=optimizer)
        current_x = env.x.data.squeeze(dim=-1).cpu().numpy()
        current_func_val = env.func_val.cpu().numpy()

        distance_x = ((current_x - env.x_opt) * (current_x - env.x_opt)).sum(axis=1).mean()
        distance_func_val = (current_func_val - env.f_opt).mean()

        print(f"episode {episode}, mean reward {mean_reward:.4f} distance_x {distance_x:.4f} distance_func_val {distance_func_val:.4f} opt_func {env.f_opt.mean():.4f}")

    agent.save(path=meta_model_path)

def test(meta_optimizer_path):
    if args.env == 'quadratic':
        env = QuadraticEnvironment(batch_size=args.batch_size, dimension=args.dimension)
    elif args.env == 'logistic':
        pass
    env.cuda()
    action_size = len(LR)
    state_size = 2 * args.dimension + 1
    agent = Agent(batch_size=args.batch_size, state_size=state_size, action_size=action_size, hidden_size=args.hidden_size)
    agent.cuda()
    agent.load(meta_optimizer_path)

    args.test_steps = 100
    env.reset()
    # import ipdb; ipdb.set_trace()
    meta_func_vals, meta_rewards = agent.test_episode(env=env, n_steps=args.test_steps, batch_size=args.batch_size) # list of scalars, list of scalars
    meta_func_vals = np.asarray(meta_func_vals)
    meta_rewards = np.asarray(meta_rewards)
    avg_meta_func_vals = np.mean(meta_func_vals, axis=1)
    avg_meta_rewards = np.mean(meta_rewards, axis=1)
    
    env.reset()
    base_optimizer = optim.Adam(env.all_params, lr=args.lr, eps=1e-5)
    base_func_vals, base_rewards = [], []
    for step in range(args.test_steps):
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
    avg_base_func_rewards = np.mean(base_rewards, axis=1)




    

def main():
    
    meta_model_path = 'logs/meta_model.pth'
    
    train(meta_model_path)

    test(meta_model_path)

if __name__ == "__main__":
    main()
