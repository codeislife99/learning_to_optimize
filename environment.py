from sklearn.datasets import make_classification
from scipy.linalg import qr
import numpy as np
import sys
import time
import argparse
from numpy.linalg import inv

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
        # print(self.current_iterate)
        func_val = .5 * np.einsum('ij,ijk,ik->i', self.current_iterate, self.H, self.current_iterate) + np.einsum('ij,ij->i', self.current_iterate, self.g)
        self.gradient = np.einsum('ijk,ik->ij',self.H, self.current_iterate) + self.g 
        reward = self.func_val - func_val
        # reward = -self.func_val + self.opti_func_val

        diff = np.sum(func_val - self.opti_func_val)/self.batch_size
        diff_x_squared = np.sum((self.opti_x.reshape(self.batch_size,self.dimensions) - self.current_iterate)* (self.opti_x.reshape(self.batch_size,self.dimensions) - self.current_iterate))/ self.batch_size
        diff_arr = func_val - self.opti_func_val
        self.func_val = func_val     
        return np.hstack((self.current_iterate, self.gradient, np.clip(np.expand_dims(self.func_val,1), -1e4, 1e4))), np.clip(reward, -1e4, 1e4),diff,diff_x_squared,diff_arr


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
