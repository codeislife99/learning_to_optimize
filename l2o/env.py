#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.linalg import qr
import pdb


LR = torch.FloatTensor([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1])


def _convert_to_param(ndarray):
    """
    converts specified numpy array to torch parameter
    """
    ndarray = ndarray.astype("float32")
    ndarray = torch.from_numpy(ndarray)
    ndarray = nn.Parameter(ndarray, requires_grad=False)
    return ndarray


class QuadraticEnvironment(nn.Module):

    def __init__(self, batch_size, dimension):
        super(QuadraticEnvironment, self).__init__()
        self.batch_size = batch_size
        self.dimension = dimension
        # H: [batch_size, dimension, dimension]
        H = np.asarray([self._generate_psd(dimension) for _ in range(batch_size)])
        self.H = _convert_to_param(H)
        # g: [batch_size, dimension, 1]
        g = np.asarray([np.random.rand(dimension) for _ in range(batch_size)])
        self.g = _convert_to_param(g.reshape(batch_size, dimension, 1))
        # Optimal point and function value
        self.x_opt = -np.einsum('ijk,ik->ij', inv(H), g) # pylint: disable=E1130
        self.f_opt = .5 * np.einsum('ij,ijk,ik->i', self.x_opt, H, self.x_opt) \
                   + np.einsum('ij,ij->i', self.x_opt, g)
        self.x = None
        self.all_params = None
        self.func_val = None

    @staticmethod
    def _generate_psd(dimension):
        A = np.random.rand(dimension, dimension)
        B = np.dot(A, A.T) / (dimension * 10) + 0.1 * np.eye(dimension, dimension)
        return B

    @staticmethod
    def _generate_psd2(dimension):
        H = np.random.randn(dimension, dimension)
        Q, _ = qr(H)
        v = np.random.rand(dimension)
        v = np.sort(v)
        v[-1] = np.random.choice([1e-1, 1e0, 1e1])
        B = np.dot(Q * v, Q.T)
        return B

    def reset(self):
        """
        reinitializes current parameter (current iterate), computes current function value and returns the state
        """
        # x: [batch_size, dimension, 1]
        x = np.random.normal(0.0, 1.0, size=(self.batch_size, self.dimension, 1))
        x = torch.from_numpy(x.astype("float32"))
        self.x = nn.Parameter(x, requires_grad=True)
        self.all_params = list(p for p in self.parameters() if p.requires_grad)
        self.func_val = self._eval()
        #print("# of weights:", len(self.all_params))
        #print("# of parameters:", sum(x.numel() for x in self.all_params))
        return self._get_state()

    def step(self, step_size): # pylint: disable=W0221
        """
        makes an iteration step in the optimization process.
        
        Parameter:
            step_size [integer] -- index into LR tensor
        Returns:
            2D torch.Tensor, 1D torch.Tensor batch, _, _ -- state, improvement, _, _

        """
        step_size = LR.gather(0, step_size)
        step_size = step_size.unsqueeze(dim=-1).unsqueeze(dim=-1)
        self.x.data.add_(-step_size * self.x.grad.data)
        next_func_val = self._eval()
        improvement = self.func_val - next_func_val
        self.func_val = next_func_val
        return self._get_state(), improvement, False, None

    def forward(self, *x):
        raise NotImplementedError

    def _eval(self):
        """
        evaluates the quadratic function at the current iterate self.x and computes the derivatives (equal to forward and backward pass).

        Returns:
             torch.Tensor -- batched objective function values
        """
        # H: [batch_size, dimension, dimension]
        H = self.H
        # g: [batch_size, dimension, 1]
        g = self.g
        # x: [batch_size, dimension, 1]
        x = self.x
        # 0.5 * x^T * H * x + g^T * x
        result = torch.bmm(torch.bmm(x.transpose(1, 2), H), x).squeeze(dim=-1).squeeze(dim=-1) * 0.5 \
               + (g * x).squeeze(dim=-1).sum(dim=-1)
        assert len(result.shape) == 1
        self._zero_grad()
        result.mean().backward()
        return result.data

    def _zero_grad(self):
        for param in self.parameters():
            if param.requires_grad is False:
                continue
            if param.grad is None:
                param.grad = Variable(param.data.new(*param.shape))
            param.grad.data.zero_()

    def _get_state(self):
        """Get the current state for the environment.

        Returns:
            2D torch.Tensor -- The current state of the environment, including parameters(current iterate), gradients, and current value of the function
        """
        forward = []
        backward = []
        for param in self.all_params:
            forward.append(param.data.view(self.batch_size, -1))
            backward.append(param.grad.data.view(self.batch_size, -1))
        result = forward + backward + [self.func_val.view(self.batch_size, -1)]
        result = torch.cat(result, dim=1)
        return result
