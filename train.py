
from agent import Agent
from environment import QuadraticEnvironment
import numpy as np
import hickle as pickle
import cProfile
from tqdm import trange
import torch.optim as optim
import torch
from torch.autograd import Variable
import sys
import time
import argparse
import pdb


class Trainer(object):
    def __init__(self):
        self.batch_size = 256
        self.dimensions = 100
        self.hidden_size = 10
        self.num_episodes = 10000

        # set of learning rates from which agent chooses from
        self.step_size_map = np.array*[10**i for i in range(-6, 1)]
        self.action_space_size = len(self.step_size_map)

        self.state_space_size = 2 * self.dimensions + 1


        self.agent = Agent(state_space=self.state_space_size, action_space_size=self.action_space_size, hidden_size=self.hidden_size)

        self.optimizer = optim.Adam([{'params': self.agent.policy_step}, {'params': self.agent.projection}], lr=0.0001)
        self.seq_length = 100

        def initialize(self):
            self.state = Variable(torch.FloatTensor(self.batch_size, self.state_space_size))
            self.memory = Variable(torch.zeros(2, self.batch_size, self.hidden_size)) # hidden state, and cell state
            self.state_seq = Variable(torch.FloatTensor(self.seq_length, self.batch_size, self.state_space_size))
            self.action_seq = Variable(torch.FloatTensor(self.seq_length, self.batch_size))
            self.reward_seq = Variable(torch.FloatTensor(self.seq_length, self.batch_size))

            
       

        def train_agent(self, state_seq, action_seq, reward_seq):
            self.optimizer.zero_grad()
            self.memory.data.zero_()
            policy_loss = self.agent.full_seq_loss(state_seq, action_seq, reward_seq, self.memory)
            reinforce_loss = policy_loss.mean()
            reinforce_loss.backward()
            self.optimizer.step()
            return reinforce_loss.data.numpy()





        def fit(self):

            env = QuadraticEnvironment(batch_size=self.batch_size, dimensions=self.dimensions)
            for episode in range(self.num_episodes):
                env.reset_state()
                state_history, action_history, reward_history = [], [], []

                total_reward = 0.0
                current_state = env.get_state()
                self.memory.data.zero_()
                # generate state action sequence using current policy by evaluating the model
                for t in trange(self.seq_length):
                    self.state.data.copy_(current_state)
                    next_action, self.memory = self.agend.fp(current_state=self.state, memory=self.memory)
                    next_action = next_action.data.numpy()
                    state_history.append(current_state)
                    action_history.append(next_action)
                    current_state, current_reward = env(self.step_size_map[next_action])
                    total_reward += current_reward
                    reward_history.append(current_reward)
                state_history = np.stack(state_history)
                action_history = np.stack(action_history)
                reward_history = np.stack(reward_history)

                self.state_seq.data.copy_(state_history)
                self.action_seq.data.copy_(action_history)
                self.reward_seq.data.copy_(reward_history)

                self.train_agent(state_history, action_history, reward_history)












