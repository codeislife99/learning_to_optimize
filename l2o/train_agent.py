#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp

import numpy as np
import torch.optim as optim

from l2o.agent import Agent, _to_cpu
from l2o.args import args
from l2o.env import LR, QuadraticEnvironment, LogisticEnvironment, MlpEnvironment
from l2o.utils import plot_data, create_dir
from l2o.utils import set_up_output_dir


from tqdm import trange

logger = None


def fix_random_seed(seed):
    import random, scipy, numpy as np, torch # pylint: disable=C0410
    random.seed(seed)
    np.random.seed(seed)
    scipy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train(meta_model_dir):
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
    agent = Agent(batch_size=args.batch_size, state_size=state_size, action_size=action_size, hidden_size=args.hidden_size, gamma=args.gamma).cuda()
    env.cuda()
    agent.cuda()

    optimizer = optim.Adam(agent.all_params, lr=args.lr, eps=1e-5)

    for episode in trange(args.n_episodes):
        mean_reward = agent.train_episode(env=env, n_steps=args.n_steps, optim=optimizer)
        current_func_val = env.func_val.cpu().numpy()
        if args.env == 'quadratic':
            best_func_val = env.f_opt
            logger.info(f"Episode {episode}, mean reward {mean_reward:.4f}, mean func val {current_func_val.mean():.4f}, opt func val {best_func_val.mean():.4f}")
        else:
            logger.info(f"Episode {episode}, mean reward {mean_reward:.4f}, mean func val {current_func_val.mean():.4f}")

        agent.save(path=f"{meta_model_dir}/model.pth")



def main():
    meta_model_dir = f'{args.save_dir}/{args.env}2/lr_{args.lr}_bs_{args.batch_size}_dim_{args.dimension}_hid_{args.hidden_size}_gamma_{args.gamma}_eps_{args.n_episodes}_steps_{args.n_steps}/'
    create_dir(meta_model_dir)
    global logger
    logger = set_up_output_dir(output_dir=meta_model_dir, file_name='train.log', name=f'{__name__}_train.log')
    logger.info(f'meta_model_dir: {meta_model_dir}')
    train(meta_model_dir)

if __name__ == "__main__":
    main()
