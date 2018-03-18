
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
from pympler.tracker import SummaryTracker
import gc
from pylab import *
plt.ion()
def curve_plot(plot_arr,episode_arr,xlabel,ylabel,number):
    figure(number)
    plt.plot(episode_arr,plot_arr)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(xlabel+'_'+ylabel+'_'+'.png')
    plt.pause(0.05)

class Trainer(object):
    def __init__(self):
        self.batch_size = 256
        self.dimensions = 100
        self.hidden_size = 10
        self.num_episodes = 500

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
        grand_total_diff = 0.0 
        episode_arr = []
        reward_arr = []
        avgreward_arr = []
        loss_arr = []
        avgloss_arr = []
        diff_arr = []
        avgdiff_arr = []
        for episode in range(self.num_episodes):
            # tracker = SummaryTracker()

            env.reset_state()
            state_history, action_history, reward_history = [], [], []

            total_reward = 0.0
            current_state = env.get_state()
            self.memory[0].data.zero_()
            self.memory[1].data.zero_()
            # generate state action sequence using current policy by evaluating the model
            diff_total = 0.0
            diff_last = 0.0
            for t in range(self.seq_length):
                self.state.data.copy_(torch.from_numpy(current_state))
                next_action, self.memory = self.agent.fp(current_state=self.state, memory=self.memory)
                next_action = next_action.data.numpy()
                state_history.append(current_state)
                action_history.append(next_action)
                current_state, current_reward,diff = env(self.step_size_map[next_action])
                diff_total += diff
                diff_last = diff
                total_reward += current_reward
                reward_history.append(current_reward)
            state_history = np.stack(state_history)
            action_history = np.stack(action_history)
            reward_history = np.stack(reward_history)

            diff_total = diff_total/(self.seq_length)

            self.state_seq.data.copy_(torch.from_numpy(state_history))
            self.action_seq.data.copy_(torch.from_numpy(action_history))
            self.reward_seq.data.copy_(torch.from_numpy(reward_history))
            grand_total_reward += total_reward
            current_loss = self.train_agent(state_history, action_history, reward_history)
            # print(np.sum(current_loss))
            grand_total_loss += current_loss
            grand_total_diff += diff_total
            episode_arr.append(episode+1)
            reward_arr.append(np.sum(total_reward))
            diff_arr.append(np.sum(diff_total))
            avgreward_arr.append(np.sum(grand_total_reward)/(episode+1))
            loss_arr.append(np.sum(current_loss))
            avgloss_arr.append(np.sum(grand_total_loss)/(episode+1))
            diff_arr.append(np.sum(diff_total))
            avgdiff_arr.append(np.sum(grand_total_diff)/(episode+1))
            print("Diff_TOTAL = ", diff_total)
            print("Diff_LAST = ", diff_last)
            # curve_plot(reward_arr,episode_arr,'Episode','Reward',0)
            # curve_plot(avgreward_arr,episode_arr,'Episode','Average Reward',1)
            # curve_plot(loss_arr,episode_arr,'Episode','Loss',2)
            # curve_plot(avgloss_arr,episode_arr,'Episode','Average Loss',3)
            # curve_plot(diff_arr,episode_arr,'Episode','Diff Value',4)
            print('Training -- Episode [%d], Average Reward: %.4f, Average Loss: %.4f'
            % (episode+1, np.sum(grand_total_reward)/(episode+1), np.sum(grand_total_loss)/(episode+1)))
            
            # tracker.print_diff()
            gc.collect()    





if __name__ == '__main__':
    t = Trainer()
    t.initialize()
    t.fit()







