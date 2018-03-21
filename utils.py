
import matplotlib.pyplot as plt 
import torch

SAVE_DIR = 'logs/'

plt.ion()
def curve_plot(plot_arr,episode_arr,xlabel,ylabel,number):
    plt.figure(number)
    plt.plot(episode_arr,plot_arr)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(xlabel+'_'+ylabel+'_'+'.png')
    plt.pause(0.05)


def save_agent(agent, sequence_length, episode, dimension):
    saved_dict = {
        "policy_step": agent.policy_step.state_dict(),
        "projection": agent.projection.state_dict(),
    }
    save_path = f'{SAVE_DIR}/dim_{dimension}_seql_{sequence_length}_episode_{episode}.pth' 
    torch.save(saved_dict, save_path)

def load_agent(agent, path):
    load_dict = torch.load(path)
    agent.policy_step.load_state_dict(load_dict["policy_step"])
    agent.projection.load_state_dict(load_dict["projection"])


