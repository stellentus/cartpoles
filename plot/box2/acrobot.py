import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box2.paths_acrobot import *

def sweep_cem_uniform_baseline():
    calibration = {
        "w/o policy": ac_knnlaplace_optim_5k,
    }
    random = ac_rnd
    true = {"true": ac_true}

    show_perform = {}
    sp_run_num = {}
    template = ac_cem_uniform[0]
    for i in range(100): # best 60
        show_perform["CEM{}".format(i)] = [template.format(i)]
        sp_run_num["CEM{}".format(i)] = [template.format(i)]

    plot_compare_top(true, calibration, None, random, "totals", "../img/ac_cem_uniform_sweep",
                     load_perf=[show_perform, sp_run_num],
                     outer=30, res_scale=-1, yscale="linear", ylim=[], ylabel="Step per episode",
                     right_ax=[],
                     label_ncol=2, true_perf_label=False)

def sweep_gridsearch_uniform_baseline():
    calibration = {
        "w/o policy": ac_knnlaplace_optim_5k,
    }
    random = ac_rnd
    true = {"true": ac_true}

    show_perform = {}
    sp_run_num = {}
    template = ac_gridsearch_uniform[0]
    for i in range(100): # best 60
        show_perform["CEM{}".format(i)] = [template.format(i)]
        sp_run_num["CEM{}".format(i)] = [template.format(i)]

    plot_compare_top(true, calibration, None, random, "totals", "../img/ac_gridsearch_uniform_sweep",
                     load_perf=[show_perform, sp_run_num],
                     outer=30, res_scale=-1, yscale="linear", ylim=[], ylabel="Step per episode",
                     right_ax=[],
                     label_ncol=2, true_perf_label=False)

def top_param():
    # PLOT 1
    calibration = {
        "Calibration-KNN": ac_knnlaplace_optim_5k,
        "Calibration-NN": ac_networkscaledlaplace_optim_5k,
    }
    random = ac_rnd
    true = {"true": ac_true}
    fqi = {"FQI": ac_fqi_tc}
    show_perform = {"RS": ac_gridsearch_uniform_best}
    sp_run_num = {"RS": ac_gridsearch_uniform_best}
    plot_compare_top(true, calibration, fqi, random, "totals", "../img/ac_plot1",
                     load_perf=[show_perform, sp_run_num],
                     outer=30, res_scale=1, ylim=[], ylabel="Return per episode",
                     right_ax=["Calibration-NN", "FQI", "Random"],
                     label_ncol=1)

    # calibration = {
    #     "w/o policy": ac_knnlaplace_suboptim_500,
    # }
    # true = {"true": ac_true}
    # cem = {"Calibration (CEM)": ac_cemlaplace}
    #
    # show_perform = {"CEM": ac_cem_uniform_best}
    # sp_run_num = {"CEM": ac_cem_uniform_best}
    #
    # plot_compare_top(true, calibration, None, [], "totals", "../img/ac_cem", cem=cem,
    #                  load_perf=[show_perform, sp_run_num],
    #                  outer=30, res_scale=-1, yscale="linear", ylim=[], ylabel="Step per episode",
    #                  right_ax=[],
    #                  label_ncol=2, true_perf_label=False)

if __name__ == '__main__':
    ranges = [0]
    # sweep_cem_uniform_baseline()
    # sweep_gridsearch_uniform_baseline()
    top_param()