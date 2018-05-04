#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from tqdm import trange

base_line = np.array([0], dtype=np.float)
base_line_decay = 0.9
from l2o.args import args

def _to_cpu(x):
    result = x
    if isinstance(result, Variable):
        result = result.data
    if result.is_cuda:
        result = result.cpu()
    return result.numpy()

def _to_scalar(tensor):
    return np.asscalar(tensor.cpu().numpy())


def _convert_to_var(x):
    result = x

    result = np.asarray(result)
    if isinstance(result, (np.ndarray, np.generic)):
        result = torch.from_numpy(result)
    if not result.is_cuda:
        result = result.cuda()
    if not isinstance(result, Variable):
        result = Variable(result, requires_grad=False, volatile=False).float()
    return result



def _optim_step(optim, rewards, log_probs, gamma):

    def _get_decayed_rewards(rewards):
        decayed_r = 0.0
        decayed_rewards = []
        for r in rewards[::-1]:
            decayed_r = r + gamma * decayed_r
            decayed_rewards.append(decayed_r)
        return list(reversed(decayed_rewards))


    rewards = _get_decayed_rewards(rewards)
    rewards = _convert_to_var(torch.stack(rewards, dim=0))
    rewards = (rewards - rewards.mean(dim=0, keepdim=True)) / (rewards.std(dim=0, keepdim=True) + 1e-6)
    log_probs = torch.stack(log_probs, dim=0)
    assert rewards.shape == log_probs.shape



    # global base_line
    # adv = (rewards - _convert_to_var(base_line)) 
    # base_line = (base_line_decay * base_line + (1 - base_line_decay) * _to_cpu(rewards.mean()))/ (1-base_line_decay)
    # loss = (adv * -log_probs).sum()

    loss = (rewards * -log_probs).sum()
    pyloss = loss.data[0]
    optim.zero_grad()
    loss.backward()
    optim.step()
    return pyloss


class PolicyStep(nn.Module):

    def __init__(self, state_size, hidden_size, dimension, project=None):
        super(PolicyStep, self).__init__()
        if project is None:
            self.policy_step = nn.LSTMCell(state_size, hidden_size)
        else:
            self.policy_step = nn.LSTMCell(2 * project.shape[-1] + 1, hidden_size)
        self.dimension = dimension
        if project is not None:
            self.project = nn.Parameter(project, requires_grad=False)

    def forward(self, state, memory): # pylint: disable=W0221
        if self.project is not None:
            # [mbs, 1, dimension]
            split_1 = state[ :, 0: self.dimension].unsqueeze(dim=1)
            split_2 = state[ :, self.dimension: self.dimension * 2].unsqueeze(dim=1)
            split_3 = state[ :, self.dimension * 2: ]
            # [mbs, steps, k]
            split_1 = torch.bmm(split_1, self.project).squeeze(dim=1)
            split_2 = torch.bmm(split_2, self.project).squeeze(dim=1)
            state = torch.cat([split_1, split_2, split_3], dim=1)
        return self.policy_step(state, memory)


class Agent(nn.Module):

    def __init__(self, batch_size, state_size, action_size, hidden_size, project=None):
        super(Agent, self).__init__()
        self.batch_size = batch_size
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.policy_step = PolicyStep(state_size, hidden_size, args.dimension, project=project)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1),
        )
        self.all_params = list(p for p in self.policy_step.parameters() if p.requires_grad) \
                        + list(self.action_head.parameters())

    def forward(self, *x):
        raise NotImplementedError

    def train_episode(self, env, n_steps, optim):
        new_tensor = self.action_head[0].weight.data.new
        memory = (
            _convert_to_var(new_tensor(self.batch_size, self.hidden_size).zero_()),
            _convert_to_var(new_tensor(self.batch_size, self.hidden_size).zero_()),
        )
        rewards, log_probs = [], []
        state = env.reset()
        for _ in range(n_steps):
            state = Variable(state.view(self.batch_size, -1).cuda(), requires_grad=False, volatile=False)
            # Get action distribution
            memory = self.policy_step(state, memory)
            action_probs = self.action_head(memory[0])
            # Sample an action
            action_cats = torch.distributions.Categorical(action_probs)
            action = action_cats.sample()
            log_prob = action_cats.log_prob(action)
            # Do one step
            state, reward, _, _ = env.step(action.data)
            # Book keeping
            rewards.append(reward)
            log_probs.append(log_prob)
        _optim_step(optim, rewards, log_probs, gamma=self.gamma)
        return torch.stack(rewards).sum(dim=1).mean()

    def test_episode(self, env, n_steps, batch_size=1):
        """
        Returns:
            list of scalars, list of scalars -- function values over n_steps, rewards
        """
        new_tensor = self.action_head[0].weight.data.new
        memory = (
            _convert_to_var(new_tensor(batch_size, self.hidden_size).zero_()),
            _convert_to_var(new_tensor(batch_size, self.hidden_size).zero_()),
        )
        func_vals, rewards = [], []
        state = env.reset()
        for _ in trange(n_steps):
            state = Variable(state.cuda(), requires_grad=False, volatile=True)
            memory = self.policy_step(state, memory)
            action_probs = self.action_head(memory[0])
            _, action = action_probs.max(dim=-1)
            action = action.data
            state, reward, _, _ = env.step(action)
            func_vals.append(_to_cpu(env.func_val))
            rewards.append(_to_cpu(reward))
        return func_vals, rewards

    def load(self, path):
        self.load_state_dict(torch.load(path)) #, map_location="cpu"))

    def save(self, path):
        torch.save(self.state_dict(), path)
