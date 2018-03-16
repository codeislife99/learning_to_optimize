
from agent import Agent
from environment import QuadraticEnvironment
import numpy as np
import pickle
import cProfile
from tqdm import trange
import torch.optim as optim
import torch
from torch.autograd import Variable
import sys
import time
import argparse
import pdb
import matplotlib.pyplot as plt 

class Trainer(object):
    def __init__(self):
        self.batch_size = 256
        self.dimensions = 100
        self.hidden_size = 10
        self.num_episodes = 10000

        # set of learning rates from which agent chooses from
        self.step_size_map = np.array([10**i for i in range(-6, 1)])
        self.action_space_size = len(self.step_size_map)

        self.state_space_size = 2 * self.dimensions + 1


        self.agent = Agent(state_space_size=self.state_space_size, action_space_size=self.action_space_size, hidden_size=self.hidden_size)

        self.optimizer = optim.SGD([{'params': self.agent.policy_step.parameters()}, {'params': self.agent.projection.parameters()}], lr=0.01)
        self.seq_length = 100

    def initialize(self):
        self.state = Variable(torch.FloatTensor(self.batch_size, self.state_space_size))
        self.memory = (
            Variable(torch.zeros(self.batch_size, self.hidden_size)), # hidden state
            Variable(torch.zeros(self.batch_size, self.hidden_size)), # cell state
        )
        self.init_memory = (
            Variable(torch.zeros(self.batch_size, self.hidden_size)), # hidden state
            Variable(torch.zeros(self.batch_size, self.hidden_size)), # cell state
        )
        self.state_seq = Variable(torch.FloatTensor(self.seq_length, self.batch_size, self.state_space_size))
        self.action_seq = Variable(torch.FloatTensor(self.seq_length, self.batch_size))
        self.reward_seq = Variable(torch.FloatTensor(self.seq_length, self.batch_size))

    def train_agent(self, state_seq, action_seq, reward_seq):
        self.optimizer.zero_grad()
        policy_loss = self.agent.full_seq_loss(state_seq, action_seq, reward_seq, self.init_memory)
        reinforce_loss = policy_loss.mean()
        reinforce_loss.backward()
        self.optimizer.step()
        return reinforce_loss.data.numpy()





    def fit(self):

        env = QuadraticEnvironment(batch_size=self.batch_size, dimensions=self.dimensions)
        grand_total_reward = 0.0 
        grand_total_loss = 0.0 
        for episode in range(self.num_episodes):
            print("episide =", episode)
            env.reset_state()
            state_history, action_history, reward_history = [], [], []

            total_reward = 0.0
            current_state = env.get_state()
            self.memory[0].data.zero_()
            self.memory[1].data.zero_()
            # generate state action sequence using current policy by evaluating the model

            for t in trange(self.seq_length):
                self.state.data.copy_(torch.from_numpy(current_state))
                next_action, self.memory = self.agent.fp(current_state=self.state, memory=self.memory)
                next_action = next_action.data.numpy()
                state_history.append(current_state)
                action_history.append(next_action)
                current_state, current_reward = env(self.step_size_map[next_action])
                total_reward += current_reward
                reward_history.append(current_reward)
            state_history = np.stack(state_history)
            action_history = np.stack(action_history)
            reward_history = np.stack(reward_history)

            self.state_seq.data.copy_(torch.from_numpy(state_history))
            self.action_seq.data.copy_(torch.from_numpy(action_history))
            self.reward_seq.data.copy_(torch.from_numpy(reward_history))
            grand_total_reward += total_reward
            grand_total_loss += self.train_agent(state_history, action_history, reward_history)
            print('\t At Episode {0:10d} Average Reward: {1:10.4f} Average Loss:{1:10.4f}'.format(episode, np.sum(grand_total_reward)/(episode+1), 
            																			np.sum(grand_total_loss)/(episode+1)))
            

                    




if __name__ == '__main__':
    t = Trainer()
    t.initialize()
    t.fit()







