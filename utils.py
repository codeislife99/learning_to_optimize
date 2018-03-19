
import matplotlib.pyplot as plt 


plt.ion()
def curve_plot(plot_arr,episode_arr,xlabel,ylabel,number):
    plt.figure(number)
    plt.plot(episode_arr,plot_arr)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(xlabel+'_'+ylabel+'_'+'.png')
    plt.pause(0.05)