import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from numpy.linalg import inv
from scipy.linalg import qr
from sklearn.datasets import make_classification


class QuadraticEnvironment(object):

    def __init__(self, batch_size, dimensions):
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.H = np.asarray([self.generate_psd2(dimensions) for _ in range(batch_size)])

        self.g = np.asarray([np.random.rand(dimensions) for _ in range(batch_size)])
        self.current_iterate = np.ones((batch_size, dimensions))
        self.func_val = .5 * np.einsum('ij,ijk,ik->i', self.current_iterate, self.H, self.current_iterate) + np.einsum('ij,ij->i', self.current_iterate, self.g)
        self.opti_x = -np.einsum('ijk,ik->ij', inv(self.H), self.g)
        self.opti_func_val = .5 * np.einsum('ij,ijk,ik->i', self.opti_x, self.H, self.opti_x) + np.einsum('ij,ij->i', self.opti_x, self.g)

        self.gradient = np.einsum('ijk,ik->ij',self.H, self.current_iterate) + self.g

    def generate_psd(self, dimensions):
        A = np.random.rand(dimensions, dimensions)
        B = np.dot(A,A.T)/(self.dimensions*10) + 0.1 * np.eye(self.dimensions,self.dimensions)
        return B

    def generate_psd2(self, dimensions):
        H = np.random.randn(dimensions, dimensions)
        Q, _ = qr(H)
        v = np.random.rand(dimensions)
        v = np.sort(v)
        v[-1] = np.random.choice([1e-1,1e0,1e1,1e2])
        B = np.dot(Q*v, Q.T)
        return B
    def get_state(self):
        return np.hstack((self.current_iterate, self.gradient, np.clip(np.expand_dims(self.func_val,1), -1e4, 1e4)))

    def reset_state(self):
        self.current_iterate = np.ones((self.batch_size, self.dimensions))
        self.func_val = .5 * np.einsum('ij,ijk,ik->i', self.current_iterate, self.H, self.current_iterate) + np.einsum('ij,ij->i', self.current_iterate, self.g)
        self.gradient = np.einsum('ijk,ik->ij',self.H, self.current_iterate) + self.g

    def __call__(self, step_size):
        '''
        Args:
            action: step size which is [B]
        Returns:
            iterate: [BxD]
            gradient: [BxD]
            func_val: [B]
            reward: [B]
        '''
        self.current_iterate = self.current_iterate - (self.gradient.T*step_size).T
        func_val = .5 * np.einsum('ij,ijk,ik->i', self.current_iterate, self.H, self.current_iterate) + np.einsum('ij,ij->i', self.current_iterate, self.g)
        self.gradient = np.einsum('ijk,ik->ij',self.H, self.current_iterate) + self.g 
        reward = self.func_val - func_val
        diff = np.sum(func_val - self.opti_func_val)/self.batch_size
        diff_x_squared = np.sum((self.opti_x.reshape(self.batch_size,self.dimensions) - self.current_iterate)* (self.opti_x.reshape(self.batch_size,self.dimensions) - self.current_iterate))/ self.batch_size
        self.func_val = func_val     
        return np.hstack((self.current_iterate, self.gradient, np.clip(np.expand_dims(self.func_val,1), -1e4, 1e4))), np.clip(reward, -1e4, 1e4),diff,diff_x_squared


class LogisticEnvironment(object):
    def __init__(self, batch_size, dimensions, sample_size=1000):
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.sample_size = sample_size
        self.current_iterate = 0.5*np.ones((batch_size, dimensions))
        self.lamda = np.random.choice([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
        self.mu = np.random.choice([0.1,0.5,1.0,2.0])
        self.X, self.Y = [], []
        for i in range(batch_size):
            X, Y = make_classification(n_samples=sample_size, n_features=dimensions-1, n_informative=dimensions-11, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=5, weights=None, flip_y=0.1, class_sep=self.mu)
            X = np.hstack((np.ones((sample_size,1)), X))
            self.X.append(X)
            self.Y.append(Y)
        self.X = np.ascontiguousarray(np.stack(self.X))
        self.Y = np.ascontiguousarray(np.stack(self.Y))
        # Initial Function values and gradients
        logits = np.einsum('ijk,ik -> ij', self.X, self.current_iterate)
        labeled_logits = self.Y*logits
        exp_logits = np.maximum(labeled_logits, 0) + np.log(1 + np.exp(-np.abs(labeled_logits)))
        self.func_val = np.mean(exp_logits, axis=1) + 0.5*self.lamda*np.einsum('ij,ij->i',self.current_iterate,self.current_iterate)
        self.gradient = np.mean((self.Y.T*self.X.T/(1+np.exp(-labeled_logits)).T).T,axis=1) + self.lamda*self.current_iterate

    def get_state(self):
        return np.hstack((self.current_iterate, self.gradient, np.clip(np.expand_dims(self.func_val,1), -1e4, 1e4)))

    def reset_state(self):
        #self.current_iterate = 0.5*np.ones((self.batch_size, self.dimensions))
        self.current_iterate -= self.current_iterate + 0.5
        logits = np.einsum('ijk,ik -> ij', self.X, self.current_iterate)
        labeled_logits = self.Y*logits
        exp_logits = np.maximum(labeled_logits, 0) + np.log(1 + np.exp(-np.abs(labeled_logits)))
        self.func_val = np.mean(exp_logits, axis=1) + 0.5*self.lamda*np.einsum('ij,ij->i',self.current_iterate,self.current_iterate)
        self.gradient = np.mean((self.Y.T*self.X.T/(1+np.exp(-labeled_logits)).T).T, axis=1) + self.lamda*self.current_iterate

    def __call__(self, step_size):
        '''
        Args:
            action: step size which is [B]
        Returns:
            iterate: [BxD]
            gradient: [BxD]
            func_val: [B]
            reward: [B]
        '''
        mini_batch_size=50
        idx = np.random.choice(self.sample_size, size=mini_batch_size, replace=False)
        mini_batchX = self.X[:,idx]
        mini_batchY = self.Y[:,idx]

        logits = np.einsum('ijk,ik -> ij', mini_batchX, self.current_iterate)
        labeled_logits = mini_batchY*logits
        exp_logits = np.maximum(labeled_logits, 0) + np.log(1 + np.exp(-np.abs(labeled_logits)))
        prev_func_val = np.mean(exp_logits, axis=1) + 0.5*self.lamda*np.einsum('ij,ij->i',self.current_iterate,self.current_iterate)

        self.current_iterate = self.current_iterate - (self.gradient.T*step_size).T
        logits = np.einsum('ijk,ik -> ij', mini_batchX, self.current_iterate)
        labeled_logits = mini_batchY*logits
        exp_logits = np.maximum(labeled_logits, 0) + np.log(1 + np.exp(-np.abs(labeled_logits)))
        func_val = np.mean(exp_logits, axis=1) + 0.5*self.lamda*np.einsum('ij,ij->i',self.current_iterate,self.current_iterate)
        self.gradient = np.mean((mini_batchY.T*mini_batchX.T/(1+np.exp(-labeled_logits)).T).T, axis=1) + self.lamda*self.current_iterate
        reward = prev_func_val - func_val
        self.func_val = func_val
        return np.hstack((self.current_iterate, self.gradient, np.clip(np.expand_dims(self.func_val,1), -1e4, 1e4))), reward


class _MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(_MLP, self).__init__()
        last_dim = input_dim
        layers = []
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(last_dim, hidden_dim, bias=True))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        self.param_pos = _MLP.get_param_pos(self)

    def forward(self, data_x):
        return self.layers(data_x)

    def get_loss(self, data_x, data_y):
        logits = self.__call__(data_x)
        result = nn.functional.cross_entropy(logits, data_y)
        return result

    def get_weights(self):
        return _MLP.to_param_vector(self, self.param_pos, use_grad=False)

    def get_grad(self):
        return _MLP.to_param_vector(self, self.param_pos, use_grad=True)

    @staticmethod
    def get_param_pos(model: nn.Module):
        pos = {}
        last_pos = 0
        for name, param in model.named_parameters():
            length = param.data.view(-1).size(0)
            pos[name] = (last_pos, last_pos + length)
            last_pos += length
        assert "len" not in pos
        pos["len"] = last_pos
        return pos

    @staticmethod
    def to_param_vector(model: nn.Module, pos: dict, use_grad):
        res = np.zeros(pos["len"], dtype="float32")
        for name, param in model.named_parameters():
            if use_grad:
                param = param.grad.data.view(-1)
            else:
                param = param.data.view(-1)
            st_pos, ed_pos = pos[name]
            res[st_pos: ed_pos] = param.cpu().numpy()
        res = torch.from_numpy(res)
        return res


class SimpleNeuralNetwork(object):

    def __init__(self, batch_size, dimensions, n_samples=100, hidden_dim=10):
        # TODO: definition of dimension
        from dataset import get_synthetic
        data_x, data_y = get_synthetic(n_samples, dimensions, n_classes=2)
        self.data_x = torch.from_numpy(data_x.astype("float32"))
        self.data_y = torch.from_numpy(data_y.astype("int64"))
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.hidden_dim = hidden_dim
        self.reset_state()

    def get_state(self):
        return np.hstack((self.current_iterate, self.gradient, np.clip(np.expand_dims(self.func_val, 1), -1e4, 1e4)))

    def reset_state(self):
        data_x = Variable(self.data_x, volatile=False)
        data_y = Variable(self.data_y, volatile=False)
        self.model = _MLP(self.dimensions, self.hidden_dim, 2, 3)
        self.model.zero_grad()
        loss = self.model.get_loss(data_x, data_y)
        loss.backward()
        self.current_iterate = self.model.get_weights()
        self.gradient = self.model.get_grad()
        self.func_val = loss.data[0]

    def __call__(self, step_size):
        data_x = Variable(self.data_x, volatile=False)
        data_y = Variable(self.data_y, volatile=False)
        self.model.zero_grad()
        loss = self.model.get_loss(data_x, data_y)
        loss.backward()
        for param in self.model.parameters():
            if param.grad is None:
                continue
            param.data.add_(-step_size, param.grad.data)
