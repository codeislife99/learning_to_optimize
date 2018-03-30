#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def _convert_to_var(tensor):
    return Variable(tensor, requires_grad=False)


class Agent(nn.Module):

    def __init__(self, state_size, action_size, hidden_size):
        super(Agent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.policy_step = nn.LSTMCell(state_size, hidden_size)
        self.projection = nn.Linear(hidden_size, action_size)

    def _step(self, agent_state, current_memory):
        # agent_state: agent state
        # memory: tuple(h_0, c_0) -- LSTM state
        #    h_0: [batch_size, hidden_size]
        #    c_0: [batch_size, hidden_size]
        updated_memory = self.policy_step(agent_state, current_memory)
        # y: [batch_size, action_size]
        action_logits = self.projection(updated_memory[0])
        return action_logits, updated_memory

    def full_seq_loss(self, state_seq, action_seq, reward_seq, memory):
        seq_length = len(state_seq)
        loss = 0.0
        memory = tuple(_convert_to_var(x.zero_()) for x in memory)
        for i in range(seq_length):
            current_state = _convert_to_var(state_seq[i])
            current_action = _convert_to_var(action_seq[i])
            current_reward = _convert_to_var(reward_seq[i])
            logits, memory = self._step(current_state, memory)
            loss += current_reward * F.cross_entropy(logits, current_action)
        return loss

    def forward(self, agent_state, current_memory):
        action_logits, updated_memory = self.step(agent_state, current_memory)
        action_prob = F.softmax(action_logits, dim=-1)
        next_action = torch.distributions.Categorical(action_prob).sample()
        return next_action, updated_memory
