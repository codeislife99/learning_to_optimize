#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def _to_scalar(tensor):
    return np.asscalar(tensor.cpu().numpy())


def _convert_to_var(tensor):
    return Variable(tensor.cuda(), requires_grad=False, volatile=False)


def _optim_step(optim, _rewards, _log_probs, gamma):

    def _get_rewards(rewards):
        decayed_r = 0.0
        decayed_rewards = []
        for r in rewards[::-1]:
            decayed_r = r.cpu().numpy() + gamma * decayed_r
            decayed_rewards.append(decayed_r)
        decayed_rewards = list(reversed(decayed_rewards))
        return np.array(decayed_rewards, dtype="float32")

    def _get_loss(rewards, log_probs):
        rewards = rewards.astype("float32").reshape(-1)
        rewards = Variable(torch.from_numpy(rewards).cuda(), requires_grad=False)
        log_probs = torch.cat(log_probs)
        policy_loss = (-rewards * log_probs).sum()
        loss = policy_loss
        return loss

    rewards = _get_rewards(_rewards)
    loss = _get_loss(rewards, _log_probs)
    pyloss = loss.data[0]

    optim.zero_grad()
    loss.backward()
    optim.step()
    return pyloss


class Agent(nn.Module):

    def __init__(self, batch_size, state_size, action_size, hidden_size):
        super(Agent, self).__init__()
        self.batch_size = batch_size
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.policy_step = nn.LSTMCell(state_size, hidden_size)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1),
        )
        self.all_params = list(self.policy_step.parameters()) + list(self.action_head.parameters())

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
            memory = self.policy_step(state, memory)
            action_probs = self.action_head(memory[0])
            action_cats = torch.distributions.Categorical(action_probs)
            action = action_cats.sample()
            log_prob = action_cats.log_prob(action)
            # import pdb; pdb.set_trace()
            state, reward, _, _ = env.step(action.data.cpu())
            rewards.append(reward)
            log_probs.append(log_prob)
        _optim_step(optim, rewards, log_probs, gamma=0.99)
        return sum(rewards).mean()

    def test_episode(self, env, n_steps):
        new_tensor = self.action_head[0].weight.data.new
        memory = (
            _convert_to_var(new_tensor(1, self.hidden_size).zero_()),
            _convert_to_var(new_tensor(1, self.hidden_size).zero_()),
        )
        func_vals, rewards = [], []
        state = env.reset()
        for _ in range(n_steps):
            state = Variable(state.cuda(), requires_grad=False, volatile=False)
            memory = self.policy_step(state, memory)
            action_probs = self.action_head(memory[0])
            assert len(action_probs) == 2
            _, action = action_probs.max(dim=-1)
            action = action.data
            state, reward, _, _ = env.step(action)
            func_vals.append(_to_scalar(env.func_val))
            rewards.append(_to_scalar(reward))
        return func_vals, rewards

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))

    def save(self, path):
        torch.save(self.state_dict(), path)
