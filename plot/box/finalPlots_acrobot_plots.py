import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_acrobot_finalPlots import *

def top_param():
    '''
    # PLOT 1

    calibration = {
        "KNN (laplace)": ac_knnlaplace_optim_5k_plot1
    }
    random = ac_rnd
    true = {"true": ac_true}
    #fqi = {"fqi": ac_fqi}
    #cem = {"cem": ac_cem}
    plot_compare_top(true, calibration, None, random, "totals", "../img/finalPlots/acrobot/plot1/plot1", outer=30, res_scale=-1)#, ylim=[110, 200])
    '''

    # PLOT 4

    calibration = {
        "KNN (laplace)": ac_knnlaplace_optim_5k_plot4,
        "network (raw)": ac_networkscaledraw_optim_5k_plot4
    }
    random = ac_rnd
    true = {"true": ac_true}
    #fqi = {"fqi": ac_fqi}
    #cem = {"cem": ac_cem}
    plot_compare_top(true, calibration, None, random, "totals", "../img/finalPlots/acrobot/plot4/plot4_scaled_raw", outer=30, res_scale=-1)#, ylim=[110, 200])
    

def sweep_model():
    '''
    # PLOT 2
    calibration = {
        "size = 500": ac_knnlaplace_optim_500_plot2,
        "size = 1000": ac_knnlaplace_optim_1k_plot2,
        "size = 2500": ac_knnlaplace_optim_2500_plot2,
        "size = 5000": ac_knnlaplace_optim_5k_plot2
    }
    true = {"true": ac_true}
    plot_generation(true, calibration, ranges, "totals", "../img/finalPlots/acrobot/plot2/plot2", outer=30, sparse_reward=-1, max_len=1000, res_scale=-1)
    '''

def data_density():
    datasets = {
        "15k": "../../data/hyperparam_v5/acrobot/offline_data/true_restarts/esarsa/step15k/optimalfixed_eps0/param_0/",
        "10k": "../../data/hyperparam_v5/acrobot/offline_data/true_restarts/esarsa/step10k/optimalfixed_eps0/param_0/",
        "5k": "../../data/hyperparam_v5/acrobot/offline_data/true_restarts/esarsa/step5k/optimalfixed_eps0/param_0/",
    }
    dimension = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
    }
    group = {"01": [0, 1], "23": [2, 3], "45": [4, 5]}
    key="new state"
    for i in range(10):
        run = "traces-{}".format(i)
        plot_dataset(datasets, key, dimension, group, run, "../img/data_density")

if __name__ == '__main__':
    ranges = [0]
    top_param()
    #sweep_model()
    # data_density()
    #top_param()