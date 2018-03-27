from agent import Agent
from environment import QuadraticEnvironment, LogisticEnvironment
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
import argparse


class Trainer(object):
    def __init__(self):
        self.batch_size = args.batch_size # 1
        self.dimensions = args.dimensions # 5
        self.hidden_size = args.hidden_size #10 
        self.num_episodes = args.num_episodes # 100
        self.seq_length = args.seq_length # 15000

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
        self.optimizer = optim.Adam(params, lr= args.lr)#, momentum=0.9)
        

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

    def train_agent(self, state_seq, action_seq, reward_seq):
        # reward_seq = (reward_seq - reward_seq.mean()) / (reward_seq.std() + np.finfo(np.float32).eps)
        self.optimizer.zero_grad()
        policy_loss = self.agent.full_seq_loss(state_seq, action_seq, reward_seq, self.init_memory)
        reinforce_loss = policy_loss.mean()
        reinforce_loss.backward()
        self.optimizer.step()
        return reinforce_loss.data.numpy()


    def fit(self):
        """
        TRAINING
        """

        env = QuadraticEnvironment(batch_size=self.batch_size, dimensions=self.dimensions)
        # env = LogisticEnvironment(batch_size=self.batch_size, dimensions=self.dimensions)
        grand_total_reward = 0.0 
        grand_total_loss = 0.0
        grand_total_diff = 0.0 
        episode_arr = []
        reward_arr = []
        acc_reward_arr = []
        final_reward_arr = []
        avgreward_arr = []
        loss_arr = []
        avgloss_arr = []
        diff_arr = []
        diff_x_arr = []
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
                    #     utils.curve_plot(reward_arr,seqs,'Iterations after final episode','Reward',0)
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
            diff_total = diff_total.sum() /(self.seq_length)
            total_reward = total_reward.sum()
            current_loss = current_loss.sum()
            diff_arr_last = diff_arr_last.sum()
            diff_x = diff_x.sum()



            grand_total_reward += total_reward
            grand_total_loss += current_loss
            grand_total_diff += diff_total




            episode_arr.append(episode+1)
            diff_arr.append(diff_total)
            loss_arr.append(current_loss)
            acc_reward_arr.append(total_reward)
            diff_x_arr.append(diff_x)



                
            # print("Diff Arr Last = ",diff_arr_last)
            print('Training -- Episode [%d], Reward: %.4f, Last Reward : %.4f, Loss: %.4f, Diff Last: %.4f,Diff X Last: %.4f'
            % (episode_arr[-1], total_reward, reward_history[-1][-1],loss_arr[-1], diff_last, diff_x_last))
            
            # tracker.print_diff()
            gc.collect()    
            episode += 1

            if episode % 20 == 0:
                utils.save_agent(agent=self.agent, dimension=self.dimensions, episode=episode, sequence_length=self.seq_length)
        

        if args.plot:
            utils.curve_plot(diff_arr,episode_arr,'Episode','Diff Value',1)
            utils.curve_plot(loss_arr,episode_arr,'Episode','Loss',2)
            utils.curve_plot(acc_reward_arr,episode_arr,'Episode','Reward',3)
            utils.curve_plot(diff_x_arr,episode_arr,'Episode','Diff X',4)

    def fit_without_rl(self, env):
        batch_size = 1
        param = Variable(torch.zeros(batch_size, self.dimensions), requires_grad=True)            
        optimizer = optim.Adam([{'params': param}], lr=0.1)

        env.reset_state()
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

            reward = reward.sum()
            val = env.func_val.sum()

            if iter % 200 == 0:
                iter_arr.append(iter)
                reward_arr.append(reward)
                diff_to_optim_val_arr.append(diff_to_optim_val)
                diff_to_optim_x_squared_arr.append(diff_to_optim_x_squared)
                function_val_arr.append(val)
        utils.curve_plot(reward_arr,iter_arr,'iter','Reward',0,label='adam')
        utils.curve_plot(diff_to_optim_val_arr,iter_arr,'iter','diff',1, label='adam')
        utils.curve_plot(diff_to_optim_x_squared_arr,iter_arr,'iter','diff_x',2, label='adam')
         
        utils.curve_plot(function_val_arr, iter_arr, 'Iterations', 'Function Value', 3, label='adam')
            # print('Training -- iter [%d], Reward: %.4f, diff: %.4f,Diff_x: %.4f, Val: %.4f' % (iter+1, reward, diff_to_optim_val, diff_to_optim_x_squared, val))
        
        param = Variable(torch.zeros(batch_size, self.dimensions), requires_grad=True)    
        optimizer = optim.SGD([{'params': param}], lr=0.1)# , momentum=0.9, weight_decay=0.9)
        env.reset_state()
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
            try:
                # print("YOLO")
                # print(param.data.numpy())
                # print(env.current_iterate)
                env.current_iterate += param.data.numpy()
            except: 
                pdb.set_trace()

            reward = reward.sum()
            val = env.func_val.sum()

            if iter % 200 == 0:
                iter_arr.append(iter)
                reward_arr.append(reward)
                diff_to_optim_val_arr.append(diff_to_optim_val)
                diff_to_optim_x_squared_arr.append(diff_to_optim_x_squared)
                function_val_arr.append(val)

        if args.plot:
            utils.curve_plot(reward_arr,iter_arr,'iter','Reward',0, label='sgd')
            utils.curve_plot(diff_to_optim_val_arr,iter_arr,'iter','Diff to optimal value',1, label='sgd')
            utils.curve_plot(diff_to_optim_x_squared_arr,iter_arr,'iter','Distance to optimal point',2, label='sgd')
            utils.curve_plot(function_val_arr, iter_arr, 'Iterations', 'Function Value', 3, label='sgd')



    def test(self, path):
        """
        path: model path to rl model
        """
        print('TESTING')
        utils.load_agent(agent=self.agent, path=path)

        if args.env == 'Quadratic':
            env = QuadraticEnvironment(batch_size=self.batch_size, dimensions=self.dimensions)
        elif args.env == 'Logistic':
            env = LogisticEnvironment(batch_size=self.batch_size, dimensions=self.dimensions)

        env.reset_state()


        # test rl agent
        current_state = env.get_state()
        self.memory[0].data.zero_()
        self.memory[1].data.zero_()

        iter_arr = []
        diff_arr = []
        diff_x_arr = []
        reward_arr = []
        value_arr = []

        for iter in range(self.seq_length):
            
            self.state.data.copy_(torch.from_numpy(current_state))
            next_action, self.memory = self.agent.fp(current_state=self.state, memory=self.memory)
            next_action = next_action.data.numpy()
            current_state, reward, diff_to_optim_val, diff_to_optim_x_squared,_= env(self.step_size_map[next_action])
            
            reward = np.sum(reward)
            diff_to_optim_val = np.sum(diff_to_optim_val)
            diff_to_optim_x_squared = np.sum(diff_to_optim_x_squared)

            if iter % 200 == 0:
                iter_arr.append(iter)
                diff_arr.append(diff_to_optim_val)
                diff_x_arr.append(diff_to_optim_x_squared)
                reward_arr.append(reward)
                value_arr.append(env.func_val.sum())

        utils.curve_plot(reward_arr, iter_arr, 'Iterations', 'Reward', 0, label='rl-agent')
        utils.curve_plot(diff_arr, iter_arr, 'Iterations', 'Diff', 1, label='rl-agent')
        utils.curve_plot(diff_x_arr, iter_arr, 'Iterations', 'Diff_x', 2, label='rl-agent')
        utils.curve_plot(value_arr, iter_arr, 'Iterations', 'Function Value', 3, label='rl-agent')

        # train without rl and test

        self.fit_without_rl(env=env)



parser = argparse.ArgumentParser(description='Meta Learning Project')
parser.add_argument('-b', '--batch-size',     default=1,           type=int,           help='mini-batch size (default: 1)')
parser.add_argument('-d', '--dimensions',     default=5,           type=int,           help='No of dimensions (default: 5)')
parser.add_argument('--hidden_size',    default=10,          type=int,           help='Hidden Size (default: 10)')
parser.add_argument('-e', '--num_episodes',         default=100,         type=int,           help='No of episodes (default: 100)')
parser.add_argument('-s', '--seq_length',        default=15000,       type=int,           help='Sequence Length (default: 15000)')
parser.add_argument('--lr','--learning_rate', default=0.1,         type=float,         help='initial learning rate')
parser.add_argument('--momentum',             default=0.9,         type=float,         help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4,        type=float,         help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq',           default=10,          type=int,           help='print frequency (default: 10)')
parser.add_argument('--resume',               default='',          type=str,           help='path to latest checkpoint (default: none)')
parser.add_argument('--test',                 dest='evaluate',     action='store_true',help='Test Mode')
parser.add_argument('-p',  '--plot',          dest='evaluate',     action='store_true',help='Plot Mode')
parser.add_argument('--env',               default='Quadratic',    type=str,           help='Env Type')



if __name__ == '__main__':
    args = parser.parse_args()
    t = Trainer()
    t.initialize()
    t.fit()
    # model_path = 'logs/dim_5_seql_15000_episode_100_quadratic.pth'
    if args.test:
        if os.path.isfile(args.resume):
            t.test(path=args.resume)








