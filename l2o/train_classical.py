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

batch_size = 256
dimension = 100
quad_env = QuadraticEnvironment(batch_size,dimension)
iter_arr = []
reward_arr = []
diff_to_optim_val_arr = []
diff_to_optim_x_squared_arr = []
function_val_arr = []
x = Variable(torch.zeros(batch_size, dimension), requires_grad=True)      
optimizer = optim.Adam([{'params': x}],lr = 0.1)

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', threshold = 0.1,min_lr = 0.1)
for iter in range(100000):
	if iter%100==0:
		current_val = x.data.cpu().numpy()
		opti_val = quad_env.x_opt
		dist = np.linalg.norm(current_val - opti_val)
		print("Dist = ", dist)
		# print("X = ", current_val)
		# print("Opti X = ", opti_val)
	result = quad_env(x)
	optimizer.zero_grad()
	result.sum().backward()
	optimizer.step()
	# dist = np.linalg.norm(x.data.cpu().numpy() - quad_env.x_opt)
	# scheduler.step(dist)
