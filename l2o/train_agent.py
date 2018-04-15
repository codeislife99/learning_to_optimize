#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.optim as optim

from l2o.agent import Agent
from l2o.args import args
from l2o.env import LR
from l2o.env import QuadraticEnvironment, MlpEnvironment


def fix_random_seed(seed):
    import random, scipy, numpy as np, torch # pylint: disable=C0410
    random.seed(seed)
    np.random.seed(seed)
    scipy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    # fix_random_seed(42)

    if args.env == 'quadratic':
        env = QuadraticEnvironment(batch_size=args.batch_size, dimension=args.dimension)
    elif args.env == 'logistic':
        pass
    elif args.env == 'mlp':
        env = MlpEnvironment(batch_size=args.batch_size, dimension=args.dimension)

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
        # current_x = env.x.data.squeeze(dim=-1).cpu().numpy()
        # distance_x = ((current_x - env.x_opt) * (current_x - env.x_opt)).sum(axis=1).mean()
        # distance_func_val = (current_func_val - env.f_opt).mean()
        # if episode % 1 == 0:
        #     print(f"episode {episode}, mean reward {mean_reward:.4f} distance_x {distance_x:.4f} mean_func {current_func_val.mean():.4f} opt_func {env.f_opt.mean():.4f}")


if __name__ == "__main__":
    main()
