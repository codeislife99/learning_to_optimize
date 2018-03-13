

import numpy as numpy
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time
import pdb


class Agent(object):
    def __init__(self, state_space_size, action_space_size, hidden_size):
        self.action_space_size = action_space_size
        self.hidden_size = hidden_size
        self.state_space_size = state_space_size

        self.policy_step = nn.LSTMCell(self.state_space_size, self.hidden_size)
        self.projection = nn.Linear(self.hidden_size, self.action_space_size)

    def step(self, x, state_tm1):
        """
        Arguments:
            x {Input} -- gradient input
            state_tm1 {(hidden, cell state)} -- each of size batch_size x hidden_dim

        Returns:
            action probabilites, memory state 
        """
        import ipdb; ipdb.set_trace()
        lstm_state = self.policy_step(x, state_tm1)
        y = self.projection(lstm_state[0])
        return y, lstm_state

    def full_seq_loss(self, state_seq, action_seq, reward_seq, init_memory):
        """
        returns policy loss
        """
        seq_length = len(state_seq)
        loss = 0.0
        memory = init_memory
        criterion = nn.CrossEntropyLoss()
        for i in range(seq_length):
            logits, memory = self.step(state_seq[i], memory)
            loss += reward_seq[i] * criterion(logits, action_seq[i])
        return loss



    def fp(self, current_state, memory):
        """
            forward pass
            current_state: pytorch variable
            memory: pytorch variable
        """
        output, updated_memory = self.step(current_state, memory)
        next_action = torch.multinomial(output, 1).squeeze() # action selection according to probabilities
        return next_action, updated_memory

    