#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from numpy.linalg import inv
from scipy.linalg import qr
from sklearn.datasets import make_classification
from l2o.dataset import get_synthetic


# LR = torch.FloatTensor([1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1])

LR = torch.FloatTensor([ 1e-3, 1e-2, 5e-2, 1e-1,  5e-1, 1.])

VALUE_CLIP = 1e4
NORM_CLIP = 10.0


def _convert_to_param(ndarray, dtype='float32'):
    """
    converts specified numpy array to torch parameter
    """

    ndarray = ndarray.astype(dtype)
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
        if self.H.is_cuda:
            x = x.cuda()
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
        global LR
        if self.H.is_cuda:
            LR = LR.cuda()
        step_size = LR.gather(0, step_size)
        step_size = step_size.unsqueeze(dim=-1).unsqueeze(dim=-1)
        self.x.data.add_(-step_size * self.x.grad.data)
        next_func_val = self._eval()
        improvement = (self.func_val - next_func_val).clamp_(-VALUE_CLIP, VALUE_CLIP)
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
        for p in self.all_params:
            p.data.clamp_(-VALUE_CLIP, VALUE_CLIP)
        torch.nn.utils.clip_grad_norm(self.all_params, NORM_CLIP, norm_type=2)
        result = torch.bmm(torch.bmm(x.transpose(1, 2), H), x).squeeze(dim=-1).squeeze(dim=-1) * 0.5 \
               + (g * x).squeeze(dim=-1).sum(dim=-1)
        assert len(result.shape) == 1
        self._zero_grad()
        result.sum().backward()
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

class LogisticEnvironment(nn.Module):

    def __init__(self, batch_size, dimension, sample_size=1000):
        super(LogisticEnvironment, self).__init__()
        self.batch_size = batch_size
        self.dimension = dimension
        self.sample_size = sample_size
        self.lamda = float(np.random.choice([1e-1, 1e-2, 1e-3, 1e-4, 1e-5]).astype('float32'))
        self.mu = np.random.choice([0.1,0.5,1.0,2.0])
        X, Y = self._generate_parameters(batch_size=self.batch_size, dimension=self.dimension, sample_size=self.sample_size, mu=self.mu)
        self.X = _convert_to_param(X)
        self.Y = _convert_to_param(Y)

        self.x = None
        self.all_params = None
        self.func_val = None
        # Data generation for logistic regression
        # Weight for data generation
        #self.w = self.mu*np.random.rand(batch_size, dimensions)
        # Generate Features
        #self.X = 2*np.random.randn(batch_size, sample_size, dimensions)
        # add 1 to incorporate bias terms
        #self.X[:,:,0] = 1
        # Get labels
        #logits = np.einsum('ijk,ik -> ij', self.X, self.w)
        #self.Y = 2*(1/(1 + np.exp(logits)) > np.random.rand(batch_size,sample_size)) - 1
        #idx = np.random.choice(self.sample_size, size=self.sample_size//10, replace=False)
        #self.Y[:,idx] = -1*self.Y[:,idx]
    @staticmethod
    def _generate_parameters(batch_size, dimension, sample_size, mu):
        res_X, res_Y = [], []
        for i in range(batch_size):
            X, Y = make_classification(n_samples=sample_size, n_features=dimension-1, n_informative=dimension-11, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=5, weights=None, flip_y=0.1, class_sep=mu)
            X = np.hstack((np.ones((sample_size,1)), X))
            res_X.append(X)
            res_Y.append(Y)
        res_X = np.ascontiguousarray(np.stack(res_X))
        res_Y = np.ascontiguousarray(np.stack(res_Y))
        return res_X, res_Y

    def reset(self):
        """
        reinitializes current parameter (current iterate), computes current function value and returns the state
        """
        # x: [batch_size, dimension, 1]
        x = 0.5 * np.ones((self.batch_size, self.dimension, 1)) #np.random.normal(0.0, 1.0, size=(self.batch_size, self.dimension, 1)) 
        x = torch.from_numpy(x.astype("float32"))
        if self.X.is_cuda :
            x = x.cuda()
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
        global LR
        if self.X.is_cuda:
            LR = LR.cuda()
        step_size = LR.gather(0, step_size)
        step_size = step_size.unsqueeze(dim=-1).unsqueeze(dim=-1)
        self.x.data.add_(-step_size * self.x.grad.data)
        next_func_val = self._eval()
        improvement = (self.func_val - next_func_val).clamp_(-VALUE_CLIP, VALUE_CLIP)
        self.func_val = next_func_val
        return self._get_state(), improvement, False, None

    
    def _eval(self):
        """
        evaluates the quadratic function at the current iterate self.x and computes the derivatives (equal to forward and backward pass).

        Returns:
             torch.Tensor -- batched objective function values
        """
        X = self.X
        Y = self.Y
        x = self.x
        for p in self.all_params:
            p.data.clamp_(-VALUE_CLIP, VALUE_CLIP)
        torch.nn.utils.clip_grad_norm(self.all_params, NORM_CLIP, norm_type=2)

        mini_batch_size=50
        idx = Variable(torch.from_numpy(np.random.choice(self.sample_size, size=mini_batch_size, replace=False)).type(torch.LongTensor), requires_grad=False)
        if X.is_cuda:
            idx = idx.cuda()

        mini_batchX = X.index_select(1, idx)
        mini_batchY = Y.index_select(1, idx)


        logits = torch.bmm(mini_batchX, x).squeeze(dim=-1)
        labeled_logits = mini_batchY * logits
        exp_logits = labeled_logits.max(0)[0] + torch.log(1 + torch.exp(-labeled_logits.abs()))
        result = exp_logits.mean(dim=1) + 0.5 * self.lamda * (x * x).squeeze(dim=-1).sum(dim=1)
        #pdb.set_trace()

        assert len(result.shape) == 1
        self._zero_grad()
        result.sum().backward()
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



class _MLP(nn.Module):

    def __init__(self):
        from sklearn.datasets import load_iris
        super(_MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )
        iris = load_iris()                       # input_dim = 4, output_dim = 3
        data_x, data_y = iris.data, iris.target  # pylint: disable=E1101
        self.data_x = _convert_to_param(data_x)
        self.data_y = _convert_to_param(data_y, dtype="int64")
        self.reset()

    def reset(self):
        for layer in self.net:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        self.param_pos = _MLP.get_param_pos(self)

    def forward(self, *x):
        raise NotImplementedError

    def get_loss(self):
        return nn.functional.cross_entropy(self.net(self.data_x), self.data_y)

    def get_weights(self):
        return _MLP.to_param_vector(self, self.param_pos, use_grad=False)

    def get_grad(self):
        return _MLP.to_param_vector(self, self.param_pos, use_grad=True)

    @staticmethod
    def get_param_pos(model: nn.Module):
        pos = {}
        last_pos = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            length = param.data.view(-1).size(0)
            pos[name] = (last_pos, last_pos + length)
            last_pos += length
        assert "len" not in pos
        pos["len"] = last_pos
        return pos

    @staticmethod
    def to_param_vector(model: nn.Module, pos: dict, use_grad):
        res = np.zeros(pos["len"], dtype="float32")
        if model.data_x.is_cuda:
            res = torch.cuda.FloatTensor(pos["len"])
        else:
            res = torch.FloatTensor(pos["len"])
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if use_grad:
                param = param.grad.data.view(-1)
            else:
                param = param.data.view(-1)
            st_pos, ed_pos = pos[name]
            res[st_pos: ed_pos].copy_(param)
        return res


class MlpEnvironment(nn.Module):

    def __init__(self, batch_size, dimension):
        super(MlpEnvironment, self).__init__()
        mlps = []
        for _ in range(batch_size):
            mlp = _MLP()
            assert mlp.param_pos["len"] == dimension, f"Dimension should be {mlp.param_pos['len']}"
            mlps.append(mlp)
        self.mlps = nn.ModuleList(mlps)
        self.func_val = None
        self.reset()

    def reset(self):
        for mlp in self.mlps:
            mlp.reset()
        self.all_params = []
        for param in self.parameters():
            if param.requires_grad:
                self.all_params.append(param)
        self.func_val = self._eval()
        return self._get_state()

    def step(self, step_size): # pylint: disable=W0221
        global LR
        if step_size.is_cuda:
            LR = LR.cuda()
        step_size = LR.gather(0, step_size)
        for idx, mlp in enumerate(self.mlps):
            step_size_i = step_size[idx]
            for param in mlp.parameters():
                if param.requires_grad:
                    param.data.add_(-step_size_i, param.grad.data)
        next_func_val = self._eval()
        improvement = (self.func_val - next_func_val).clamp_(-VALUE_CLIP, VALUE_CLIP)
        self.func_val = next_func_val
        return self._get_state(), improvement, False, None

    def forward(self, *x):
        raise NotImplementedError

    def _eval(self):
        losses = []
        for mlp in self.mlps:
            losses.append(mlp.get_loss())
        loss = torch.cat(losses, dim=0)
        self._zero_grad()
        loss.sum().backward()
        for x in self.parameters():
            if x.requires_grad:
                x.data.clamp_(-VALUE_CLIP, VALUE_CLIP)
        torch.nn.utils.clip_grad_norm([x for x in self.parameters() if x.requires_grad], NORM_CLIP, norm_type=2)
        return loss.data

    def _zero_grad(self):
        for param in self.parameters():
            if param.requires_grad is False:
                continue
            if param.grad is None:
                param.grad = Variable(param.data.new(*param.shape))
            param.grad.data.zero_()

    def _get_state(self):
        batch_size = len(self.mlps)
        forward, backward = [], []
        for mlp in self.mlps:
            forward.append(mlp.get_weights())
            backward.append(mlp.get_grad())
        forward = torch.stack(forward, dim=0)
        backward = torch.stack(backward, dim=0)
        result = [forward, backward, self.func_val.view(batch_size, -1)]
        result = torch.cat(result, dim=1)
        return result
