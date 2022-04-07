import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box2.paths_cartpole import *

def top_param():
    calibration = {
        "Optimal": cart_optim_knn,
        "Medium": cart_suboptim_knn,
    }
    random = cart_rnd
    true = {"true": cart_true}

    plot_compare_top(true, calibration, None, random, "cartpole-failures", "../img/cartpole",
                     outer=30, res_scale=-1, yscale="linear", ylim=[], ylabel="Step per episode",
                     label_ncol=2, true_perf_label=False)


if __name__ == '__main__':
    ranges = [0]
    top_param()
