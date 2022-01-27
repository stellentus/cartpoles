import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.check.paths_puddlerand_finalPlots import *

def top_param():

    # PLOT 1
    calibration = {
        "Calibration-KNN": pr_knnlaplace_optim_5k,
        # "Calibration (raw)": pr_knnraw_optim_5k,
        "Calibration-NN": pr_networkscaledlaplace_optim_5k,
        # "NN Calibration (raw)": pr_networkscaledraw_optim_5k,
    }
    random = pr_rnd
    true = {"true": pr_true}
    fqi = {"FQI": pr_fqi_tc}
    #cem = {"cem": pr_cem}
    # plot_compare_top(true, calibration, fqi, random, "totals", "../img/finalPlots/puddlerand/plot1/plot1_models",
    #                  outer=30, ylim=[[-100, -20], []], ylabel="Return per episode", right_ax=["FQI", "Random"],
    #                  label_ncol=6)

    # PLOT 2
    calibration = {
        "Size = 5000": pr_knnlaplace_avg_5k_new,
        # "Size = 2500": pr_knnlaplace_avg_2500_new,
        "Size = 1000": pr_knnlaplace_avg_1k_new,
        "Size = 500": pr_knnlaplace_avg_500_new,
    }
    true = {"true": pr_true}
    # plot_compare_top(true, calibration, None, [], "totals", "../img/finalPlots/puddlerand/plot2/plot2_size",
    #                  outer=30, ylim=[[-34, -27]], ylabel="Return per episode", right_ax=[],
    #                  label_ncol=3, plot="bar")

    calibration = {
        "Optimal": pr_knnlaplace_optim_5k_new,
        "Medium": pr_knnlaplace_avg_5k_new,
        "Naive": pr_knnlaplace_bad_5k_new,
    }
    # plot_compare_top(true, calibration, None, [], "totals", "../img/finalPlots/puddlerand/plot2/plot2_policy",
    #                  outer=30, ylim=[[-34, -27]], ylabel="Return per episode", right_ax=[],
    #                  label_ncol=3, plot="bar", true_perf_label=False)

    # PLOT CEM
    calibration = {
        "Calibration-KNN": pr_knnlaplace_optim_5k
    }
    true = {"true": pr_true}
    cem = {"calibration (cem)": pr_cemlaplace_optim_5k}
    # fqi = {"FQI": pr_fqi_tc}
    # plot_compare_top(true, calibration, None, random, "totals", "../img/finalPlots/puddlerand/plot1/plot1_models_CEM_KNNlaplace", cem=cem,
    #                   outer=30, ylim=[[-80, -20],[]], ylabel="Return per episode", right_ax=[], plot ='box')

    calibration = {
        "Calibration (grid search)": pr_k3_laplace_suboptim_500data
    }
    #random = pr_rnd_30
    random = []
    true = {"true": pr_true_cem}
    # true = {"true": pr_true}
    cem = {"Calibration (CEM)": pr_CEM_k3_laplace_suboptim_500data_100iters}
    #fqi = {"FQI": ac_fqi_tc}
    # plot_compare_top(true, calibration, None, random, "totals", "../img/finalPlots/puddlerand/cem/cem_k3_laplace_100iters_ylim", cem=cem,
    #                   outer=30, ylim=[[-40, -25]], ylabel="Return per episode", right_ax=[], plot ='box')


    # PLOT RAW
    calibration = {
        "KNN-laplace": pr_knnlaplace_optim_5k,
        "KNN-raw": pr_knnraw_optim_5k,
        "NN-laplace": pr_networkscaledlaplace_optim_5k,
        "NN-raw": pr_networkscaledraw_optim_5k,
    }
    random = pr_rnd
    true = {"true": pr_true}
    fqi = {"FQI": pr_fqi_tc}
    #cem = {"cem": pr_cem}
    # plot_compare_top(true, calibration, fqi, [], "totals", "../img/finalPlots/puddlerand/appendix/raw",
    #                  outer=30, ylim=[[-100, -20], []], ylabel="Return per episode", right_ax=["FQI", "Random", "NN-raw"],
    #                  label_ncol=6)

    # PLOT Agents
    # calibration = {
    #     "Esarsa": pr_knnlaplace_optim_5k,
    #     # "DQN": pr_dqn_knnlaplace_optim,
    #     "AC": pr_actorcritic_knnlaplace_optim,
    # }
    # true = {
    #     "Esarsa": pr_true,
    #     # "DQN": pr_dqn,
    #     "AC": pr_actorcritic,
    # }
    # plot_compare_agents(true, calibration, None, [], "totals", "../img/finalPlots/puddlerand/plot_agents",
    #                     outer=30, ylim=[[-50, -25]], ylabel="Return per episode", right_ax=[],
    #                     label_ncol=3)
    calibration = {
        "Calibration-KNN": pr_actorcritic_knnlaplace_optim,
    }
    random = random_generator(36)
    true = {"true": pr_actorcritic}
    fqi = {"FQI": pr_fqi_tc}
    #cem = {"cem": pr_cem}
    plot_compare_top(true, calibration, fqi, random, "totals", "../img/finalPlots/puddlerand/appendix/ac",
                     outer=30, ylim=[[-100, -20], []], ylabel="Return per episode", right_ax=["FQI", "Random"],
                     label_ncol=6)

    # PLOT FQI NN
    calibration = {
        "Calibration": pr_knnlaplace_optim_5k,
    }
    show_perform = {
        "FQI-TC": pr_fqi_tc,
        "FQI-NN": pr_fqi_nn,
    }
    true = {"true": pr_true}
    plot_compare_top(true, calibration, fqi, [], "totals", "../img/finalPlots/puddlerand/appendix/fqi",
                     load_perf=show_perform,
                     outer=30, ylim=[[-100, -20]], ylabel="Return per episode", right_ax=["FQI-NN", "FQI-TC"],
                     label_ncol=2, true_perf_label=False)


def sweep_model():
    '''
    # PLOT 2
    calibration = {
        "size = 500": pr_knnlaplace_optim_500_plot2,
        "size = 1000": pr_knnlaplace_optim_1k_plot2,
        "size = 2500": pr_knnlaplace_optim_2500_plot2,
        "size = 5000": pr_knnlaplace_optim_5k_plot2
    }
    true = {"true": pr_true}
    plot_generation(true, calibration, ranges, "totals", "../img/finalPlots/puddlerand/plot2/plot2_boxplot", outer=30, sparse_reward=-1, max_len=1000)
    '''
    '''
    # PLOT 3
    calibration = {
        "Optimal": pr_knnlaplace_optim_5k_plot3,
        "average policy": pr_knnlaplace_suboptim_5k_plot3,
        "Naive": pr_knnlaplace_subsuboptim_5k_plot3
    }
    true = {"true": pr_true}
    plot_generation(true, calibration, ranges, "totals", "../img/finalPlots/puddlerand/plot3/plot3_boxplot", outer=30, sparse_reward=-1, max_len=1000)
    '''

    '''
    # PLOT CEM

    calibration = {
        "Calibration-KNN": pr_knnraw_optim_5k_old,
        "calibration (cem)": pr_cemraw_optim_5k_old
    }
    #random = ac_rnd
    true = {"true": pr_true_old}
    #fqi = {"FQI": ac_fqi_tc}
    plot_generation(true, calibration, ranges, "totals", "../img/finalPlots/puddlerand/plot1/plot1_models_CEM", ylim=[], outer=30, sparse_reward=-1, max_len=1000)
    '''

def data_density():
    datasets = {
        "random": "../../data/hyperparam_v4/puddle/offline_data/esarsa/step10k_env/random/param_0/",
        "return -320": "../../data/hyperparam_v4/puddle/offline_data/esarsa/step10k_env/return-320/param_0/",
        "return -40": "../../data/hyperparam_v4/puddle/offline_data/esarsa/step10k_env/return-40/param_0/",
    }
    dimension = {
        0: "x",
        1: "y",
    }
    group = {"xy": [0, 1]}
    key="new state"
    for i in range(10):
        run = "traces-{}".format(i)
        plot_dataset(datasets, key, dimension, group, run, "../img/puddle_data_density")


if __name__ == '__main__':
    ranges = [0]
    top_param()
    #sweep_model()
    #data_density()
