import numpy as np 
import os.path as osp
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 


def plot_data(data_x, data_y, label_x, label_y, label, fig_no):
    fig = plt.figure(fig_no, figsize=(10, 10))
    plt.plot(data_x , data_y, label=label)
    plt.legend()
    plt.ylabel(label_y)
    plt.xlabel(label_x)
    return fig
<<<<<<< HEAD


def create_dir(directory):
    if not osp.exists(directory):
        os.makedirs(directory)
        print(f"created directory: {directory}")
    
=======
>>>>>>> b7bd3726480ed85a0c48e30b622a5c7150ad6b1e
