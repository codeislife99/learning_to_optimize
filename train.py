
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
import utils


class Trainer(object):
    def __init__(self):
        self.batch_size = 256
        self.dimensions = 100
        self.hidden_size = 10
        self.num_episodes = 500
        self.seq_length = 100

        # set of learning rates from which agent chooses from
        self.step_size_map = np.array([10**i for i in range(-6, 1)])
        self.action_space_size = len(self.step_size_map)

        self.state_space_size = 2 * self.dimensions + 1


        self.agent = Agent(state_space_size=self.state_space_size, action_space_size=self.action_space_size, hidden_size=self.hidden_size)

        self.optimizer = optim.SGD([{'params': self.agent.policy_step.parameters()}, {'params': self.agent.projection.parameters()}], lr=0.1, momentum=0.9)
        

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
            diff_x_last = 0.0
            for t in range(self.seq_length):
                self.state.data.copy_(torch.from_numpy(current_state))
                next_action, self.memory = self.agent.fp(current_state=self.state, memory=self.memory)
                
                next_action = next_action.data.numpy()
                state_history.append(current_state)
                action_history.append(next_action)
                current_state, current_reward,diff,diff_x = env(self.step_size_map[next_action])
                diff_total += diff
                diff_last = diff
                diff_x_last = diff_x
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
            # print("Diff_TOTAL = ", diff_total)
            # print("Diff_LAST = ", diff_last)
            # utils.curve_plot(reward_arr,episode_arr,'Episode','Reward',0)
            # utils.curve_plot(avgreward_arr,episode_arr,'Episode','Average Reward',1)
            # utils.curve_plot(loss_arr,episode_arr,'Episode','Loss',2)
            # utils.urve_plot(avgloss_arr,episode_arr,'Episode','Average Loss',3)
            # utils.curve_plot(diff_arr,episode_arr,'Episode','Diff Value',4)
            print('Training -- Episode [%d], Average Reward: %.4f, Average Loss: %.4f,Diff Last: %.4f,Diff X Last: %.4f'
            % (episode+1, np.sum(grand_total_reward)/(episode+1), np.sum(grand_total_loss)/(episode+1),diff_last,diff_x_last))
            
            # tracker.print_diff()
            gc.collect()    



    def fit_without_rl(self, optim_type='adam'):
        param = Variable(torch.zeros(self.batch_size, self.dimensions), requires_grad=True)
        env = QuadraticEnvironment(batch_size=self.batch_size, dimensions=self.dimensions)
        if optim_type == 'sgd':
            optimizer = optim.SGD([{'params': param}], lr=0.00001, momentum=0.9, weight_decay=0.9)
        elif optim_type == 'adam':
            optimizer = optim.Adam([{'params': param}], lr=0.001)

        episode_arr = []
        reward_arr = []
        diff_to_optim_val_arr = []
        diff_to_optim_x_arr = []
        function_val_arr = []

        for episode in range(self.num_episodes):
            state, reward, diff_to_optim_val, diff_to_optim_x = env(step_size=0.)
            current_iterate, current_gradient = state[:, :self.dimensions], state[:, self.dimensions: 2 * self.dimensions]
            
            optimizer.zero_grad()
            param.sum().backward()
            param.grad.data.copy_(torch.from_numpy(current_gradient))
            optimizer.step()
            env.current_iterate += param.data.numpy()

            episode_arr.append(episode)
            reward = reward.sum()
            reward_arr.append(reward)
            diff_to_optim_val_arr.append(diff_to_optim_val)
            diff_to_optim_x_arr.append(diff_to_optim_x)
            val = env.func_val.sum()
            function_val_arr.append(val)
            # pdb.set_trace()
            utils.curve_plot(reward_arr,episode_arr,'Episode','Reward',1)
            utils.curve_plot(diff_to_optim_val_arr,episode_arr,'Episode','diff',2)
            utils.curve_plot(diff_to_optim_x_arr,episode_arr,'Episode','diff_x',3)
            utils.curve_plot(function_val_arr,episode_arr,'Episode','F value',4)

            print('Training -- Episode [%d], Reward: %.4f, diff: %.4f,Diff_x: %.4f, Val: %.4f' % (episode+1, reward, diff_to_optim_val, diff_to_optim_x, val))
            





            
            






if __name__ == '__main__':
    t = Trainer()
    t.initialize()
    t.fit_without_rl()







