import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 


def plot_data(data_x, data_y, label_x, label_y, label, fig_no):
    fig = plt.figure(fig_no, figsize=(10, 10))
    plt.plot(data_x , data_y, label=label)
    plt.legend()
    plt.ylabel(label_y)
    plt.xlabel(label_x)
    return fig
