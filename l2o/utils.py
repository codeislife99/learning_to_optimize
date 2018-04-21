import numpy as np 
import os.path as osp
import os
import matplotlib
import logging
matplotlib.use('agg')
import matplotlib.pyplot as plt 


def plot_data(data_x, data_y, label_x, label_y, label, fig_no):
    fig = plt.figure(fig_no, figsize=(10, 10))
    plt.plot(data_x , data_y, label=label)
    plt.legend()
    plt.ylabel(label_y)
    plt.xlabel(label_x)
    return fig


def create_dir(directory):
    if not osp.exists(directory):
        os.makedirs(directory)
        print(f"created directory: {directory}")
    

def set_up_output_dir(output_dir, file_name, name, log_level='INFO'):
    
    log_file = osp.join(output_dir, file_name)


    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] [%(filename)s:%(lineno)s]  %(message)s")
    rootLogger = logging.getLogger(name)
    rootLogger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler(osp.join(output_dir, file_name))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    return rootLogger