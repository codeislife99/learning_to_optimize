#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.linalg import qr


def _convert_to_param(ndarray):
    ndarray = ndarray.astype("float32")
    ndarray = torch.from_numpy(ndarray)
    ndarray = nn.Parameter(ndarray, requires_grad=False)
    return ndarray


class QuadraticEnvironment(nn.Module):

    def __init__(self, batch_size, dimension):
        super(QuadraticEnvironment, self).__init__()
        # H: [batch_size, dimension, dimension]
        H = np.asarray([self._generate_psd2(dimension) for _ in range(batch_size)])
        self.H = _convert_to_param(H)
        # g: [batch_size, dimension, 1]
        g = np.asarray([np.random.rand(dimension) for _ in range(batch_size)])
        self.g = _convert_to_param(g.reshape(batch_size, dimension, 1))
        # Optimal point and function value
        self.x_opt = -np.einsum('ijk,ik->ij', inv(H), g) # pylint: disable=E1130
        self.f_opt = .5 * np.einsum('ij,ijk,ik->i', self.x_opt, H, self.x_opt) \
                   + np.einsum('ij,ij->i', self.x_opt, g)

    @staticmethod
    def generate_psd(dimension):
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
        B = np.dot(Q*v, Q.T)
        return B

    def forward(self, x):
        # x: [batch_size, dimension]

        # H: [batch_size, dimension, dimension]
        H = Variable(self.H, requires_grad=False)
        # g: [batch_size, dimension, 1]
        g = Variable(self.g, requires_grad=False)
        # x: [batch_size, dimension, 1]
        x = x.unsqueeze(dim=-1)
        # 0.5 * x^T * H * x + g^T * x
        result = torch.bmm(torch.bmm(x.t(), H), x) * 0.5 \
               + (g * x).squeeze(dim=-1).sum(dim=-1)
        return result
