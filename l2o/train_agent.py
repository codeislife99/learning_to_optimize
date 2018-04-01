#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from l2o.args import args
from l2o.env import QuadraticEnvironment, LR
from l2o.agent import Agent

import torch.optim as optim

def main():
    if args.env == 'quadratic':
        env = QuadraticEnvironment(batch_size=args.batch_size, dimension=args.dimension)
    elif args.env == 'logistic':
        pass

    action_size = len(LR)
    state_size = 2 * args.dimension + 1
    agent = Agent(batch_size=args.batch_size, state_size=state_size, action_size=action_size, hidden_size=args.hidden_size)

    optimizer = optim.Adam(agent.all_params, lr=args.lr, eps=1e-5)

    for episode in range(args.n_episodes):
        mean_reward = agent.train_episode(env=env, n_steps=args.n_steps, optim=optimizer)
        current_x = env.x.data.squeeze(dim=-1).cpu().numpy()
        current_func_val = env.func_val.cpu().numpy()

        distance_x = ((current_x - env.x_opt) * (current_x - env.x_opt)).sum(axis=1).mean()
        distance_func_val = (current_func_val - env.f_opt).mean()

        if episode % 100 == 0:
            print(f"episode {episode}, mean reward {mean_reward:.4f} distance_x {distance_x:.4f} distance_func_val {distance_func_val:.4f}")


if __name__ == "__main__":
    main()