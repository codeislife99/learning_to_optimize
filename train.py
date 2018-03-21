from agent import Agent
from environment import SimpleNeuralNetwork
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
# from pympler.tracker import SummaryTracker
import gc
from pylab import *
import utils


class Trainer(object):
    def __init__(self):
        self.batch_size = 1 # 256
        self.dimensions = 12 # 100
        self.hidden_size = 10 
        self.num_episodes = 500
        self.seq_length = 15000 # 100

        # set of learning rates from which agent chooses from
        # self.step_size_map = np.array([10**i for i in range(-10, -1)])        
        step_size_map = [10**i for i in range(-6, -1)]
        self.step_size_map = []
        for step in step_size_map:
            for divider in range(2,10):
                self.step_size_map.append(1.0*step/divider)

        self.step_size_map = np.array(self.step_size_map)
        # self.step_size_map = np.array([0.01])

        self.action_space_size = len(self.step_size_map)
        print("Action Space Size = ", self.action_space_size)
        self.state_space_size = 2 * self.dimensions + 1


        self.agent = Agent(state_space_size=self.state_space_size, action_space_size=self.action_space_size, hidden_size=self.hidden_size)

        # self.optimizer = optim.SGD([{'params': self.agent.policy_step.parameters()}, {'params': self.agent.projection.parameters()}], lr=10 , momentum=0.9)
        params = list(self.agent.policy_step.parameters()) + list(self.agent.projection.parameters())
        self.optimizer = optim.SGD(params, lr=0.01 )#, momentum=0.9)
        

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
        # self.state_seq = Variable(torch.FloatTensor(self.seq_length, self.batch_size, self.state_space_size))
        # self.action_seq = Variable(torch.LongTensor(self.seq_length, self.batch_size))
        # self.reward_seq = Variable(torch.FloatTensor(self.seq_length, self.batch_size))

    def train_agent(self, state_seq, action_seq, reward_seq):
        # reward_seq = (reward_seq - reward_seq.mean()) / (reward_seq.std() + np.finfo(np.float32).eps)
        self.optimizer.zero_grad()
        policy_loss = self.agent.full_seq_loss(state_seq, action_seq, reward_seq, self.init_memory)
        reinforce_loss = policy_loss.mean()
        reinforce_loss.backward()
        self.optimizer.step()
        return reinforce_loss.data.numpy()


    def fit(self):

        env = SimpleNeuralNetwork(batch_size=self.batch_size, dimensions=self.dimensions)
        grand_total_reward = 0.0 
        grand_total_loss = 0.0
        grand_total_diff = 0.0 
        episode_arr = []
        reward_arr = []
        final_reward_arr = []
        avgreward_arr = []
        loss_arr = []
        avgloss_arr = []
        diff_arr = []
        avgdiff_arr = []

        episode = 0
        while episode < self.num_episodes:
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
            diff_arr_last = 0.0 
            # reward_last = 0.0 
            try:
                seqs = []
                for t in range(self.seq_length):
                    self.state.data.copy_(torch.from_numpy(current_state))
                    next_action, self.memory = self.agent.fp(current_state=self.state, memory=self.memory)
                    
                    next_action = next_action.data.numpy()
                    state_history.append(current_state)
                    action_history.append(next_action)
                    current_state, current_reward, diff, diff_x,diff_arr_last = env(self.step_size_map[next_action])
                    # if t%100 == 0:
                    #     pass
                    #     print(current_reward)
                    reward_history.append(current_reward)

                    diff_total += diff
                    diff_last = diff
                    diff_x_last = diff_x
                    total_reward += current_reward

                    # print(reward_arr)
                    # if episode == self.num_episodes - 1 and t % 100 == 0:
                    #     reward_arr.append(np.sum(total_reward))
                    #     seqs.append(t)
                    #     utils.curve_plot(reward_arr,seqs,'Iterations','Reward',0)
            except KeyboardInterrupt:
                raise
            except:
                print("Out of Hand")
                raise
                continue
            state_history = np.stack(state_history)
            action_history = np.stack(action_history)
            reward_history = np.stack(reward_history)
            current_loss = self.train_agent(state_history, action_history, reward_history)
            diff_total = sum(diff_total) /(self.seq_length)
            total_reward = sum(total_reward)
            current_loss = sum(current_loss)



            grand_total_reward += total_reward
            grand_total_loss += current_loss
            grand_total_diff += diff_total

            episode_arr.append(episode+1)
            final_reward_arr.append(current_reward)
            diff_arr.append(diff_total)
            loss_arr.append(current_loss)
            avgreward_arr.append(grand_total_reward/(episode+1))
            avgloss_arr.append(grand_total_loss/(episode+1))
            avgdiff_arr.append(grand_total_diff/(episode+1))
            # utils.curve_plot(diff_arr,episode_arr,'Episode','Diff Value',1)
            # utils.curve_plot(loss_arr,episode_arr,'Episode','Loss',2)
            print("Diff Arr Last = ",diff_arr_last)
            print('Training -- Episode [%d], Reward: %.4f, Last Reward : %.4f, Loss: %.4f, Diff Last: %.4f,Diff X Last: %.4f'
            % (episode_arr[-1], total_reward, reward_history[-1][-1],loss_arr[-1], diff_last, diff_x_last))
            
            # tracker.print_diff()
            gc.collect()    
            episode += 1

        utils.save_agent(agent=self.agent, dimension=self.dimensions, episode=episode, sequence_length=self.seq_length)
    

    def fit_without_rl(self, optim_type='adam'):
        batch_size = 1
        param = Variable(torch.zeros(batch_size, self.dimensions), requires_grad=True)
        env = SimpleNeuralNetwork(batch_size=self.batch_size, dimensions=self.dimensions)
        if optim_type == 'sgd':
            optimizer = optim.SGD([{'params': param}], lr=0.1)# , momentum=0.9, weight_decay=0.9)
        elif optim_type == 'adam':
            optimizer = optim.Adam([{'params': param}], lr=0.01)

        iter_arr = []
        reward_arr = []
        diff_to_optim_val_arr = []
        diff_to_optim_x_squared_arr = []
        function_val_arr = []

        for iter in range(self.seq_length):
            state, reward, diff_to_optim_val, diff_to_optim_x_squared,_ = env(step_size=0.)
            current_iterate, current_gradient = state[:, :self.dimensions], state[:, self.dimensions: 2 * self.dimensions]
            
            optimizer.zero_grad()
            param.data.zero_()
            param.sum().backward()
            param.grad.data.copy_(torch.from_numpy(current_gradient))
            optimizer.step()
            env.current_iterate += param.data.numpy()

            iter_arr.append(iter)
            reward = reward.sum()
            reward_arr.append(reward)
            diff_to_optim_val_arr.append(diff_to_optim_val)
            diff_to_optim_x_squared_arr.append(diff_to_optim_x_squared)
            val = env.func_val.sum()
            function_val_arr.append(val)
            # utils.curve_plot(reward_arr,iter_arr,'iter','Reward',1)
            # utils.curve_plot(diff_to_optim_val_arr,iter_arr,'iter','diff',2)
            # utils.curve_plot(diff_to_optim_x_squared_arr,iter_arr,'iter','diff_x',3)
            # utils.curve_plot(function_val_arr,iter_arr,'iter','value',4)

            print('Training -- iter [%d], Reward: %.4f, diff: %.4f,Diff_x: %.4f, Val: %.4f' % (iter+1, reward, diff_to_optim_val, diff_to_optim_x_squared, val))
            



    def test(self, path):
        utils.load(agent=self.agent, path=path)

            
            






if __name__ == '__main__':
    t = Trainer()
    t.initialize()
    t.fit()
    model_path = 'logs/dim_5_seql_15000_episode_2.pth'
    t.test(path=model_path)

    # t.fit_without_rl()







