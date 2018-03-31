import torch.optim as optim
from env import QuadraticEnvironment
import numpy as np
import torch.optim as optim
import torch
from torch.autograd import Variable
import sys
import time
import argparse
import pdb
import matplotlib.pyplot as plt 
from pylab import *
import argparse

batch_size = 10
dimension = 5
quad_env = QuadraticEnvironment(batch_size,dimension)
iter_arr = []
reward_arr = []
diff_to_optim_val_arr = []
diff_to_optim_x_squared_arr = []
function_val_arr = []
x = Variable(torch.zeros(batch_size, dimension), requires_grad=True)      
optimizer = optim.Adam([{'params': x}],lr = 0.01)
print(quad_env.x_opt)
for iter in range(1000000):
	print("X = ", x.data.cpu().numpy())
	print("Opti X = ", quad_env.x_opt)
	result = quad_env(x)
	optimizer.zero_grad()
	result.sum().backward()
	optimizer.step()
