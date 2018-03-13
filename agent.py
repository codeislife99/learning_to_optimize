

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
    def __init__(self, state_space, action_space_size, hidden_size)
        self.action_space_size = action_space_size
        self.hidden_size = hidden_size
        self.state_space_size = state_spafe_size

        self.policy_step = nn.LSTM(self.state_space_size, self.hidden_size)
        self.projection = nn.Linear(self.hidden_size, self.action_space_size)

    def step(self, x, state_tm1):
        """
        Arguments:
            x {Input} -- gradient input
            state_tm1 {(hidden, cell state)} -- each of size batch_size x hidden_dim

        Returns:
            action probabilites, memory state 
        """
        lstm_state = self.policy_step(x, state_tm1)
        y = self.projection(lstm_state[0])
        return y, lstm_state

    