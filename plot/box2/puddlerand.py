import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box2.paths_puddlerand import *

def sweep_cem_uniform_baseline():
    calibration = {
        "w/o policy": pr_knnlaplace_suboptim_500,
    }
    random = pr_rnd
    true = {"true": pr_true}

    show_perform = {}
    sp_run_num = {}
    template = pr_cem_uniform[0]
    for i in range(100): #  best 52
        show_perform["CEM{}".format(i)] = [template.format(i)]
        sp_run_num["CEM{}".format(i)] = [template.format(i)]

    plot_compare_top(true, calibration, None, random, "totals", "../img/pr_cem_uniform_sweep",
                     load_perf=[show_perform, sp_run_num],
                     outer=30, yscale="linear", ylim=[], ylabel="Step per episode",
                     right_ax=[],
                     label_ncol=2, true_perf_label=False)

def sweep_gridsearch_uniform_baseline():
    calibration = {
        "w/o policy": pr_knnlaplace_suboptim_500,
    }
    random = pr_rnd
    true = {"true": pr_true}

    show_perform = {}
    sp_run_num = {}
    template = pr_gridsearch_uniform[0]
    for i in range(100): #  best 52
        show_perform["CEM{}".format(i)] = [template.format(i)]
        sp_run_num["CEM{}".format(i)] = [template.format(i)]

    plot_compare_top(true, calibration, None, random, "totals", "../img/pr_gridsearch_uniform_sweep",
                     load_perf=[show_perform, sp_run_num],
                     outer=30, yscale="linear", ylim=[], ylabel="Step per episode",
                     right_ax=[],
                     label_ncol=2, true_perf_label=False)

def top_param():

    calibration = {
            "Calibration-KNN": pr_knnlaplace_optim_5k,
            # "Calibration (raw)": pr_knnraw_optim_5k,
            "Calibration-NN": pr_networkscaledlaplace_optim_5k,
            # "NN Calibration (raw)": pr_networkscaledraw_optim_5k,
        }
    random = pr_rnd
    true = {"true": pr_true}
    fqi = {"FQI": pr_fqi_tc}
    show_perform = {"RS": pr_gridsearch_uniform_online}
    sp_run_num = {"RS": pr_gridsearch_uniform}
    plot_compare_top(true, calibration, fqi, random, "totals", "../img/pr_plot1",
                     load_perf=[show_perform, sp_run_num],
                     outer=30, yscale="linear", ylim=[-100, -24], ylabel="Return per episode",
                     right_ax=["FQI", "Random"],
                     label_ncol=6)


    calibration = {
        "w/o policy": pr_knnlaplace_suboptim_500,
    }
    true = {"true": pr_true}
    cem = {"Calibration (CEM)": pr_cemlaplace}
    show_perform = {"RS": pr_cem_uniform_online}
    sp_run_num = {"RS": pr_cem_uniform}
    plot_compare_top(true, calibration, None, [], "totals", "../img/pr_cem", cem=cem,
                     load_perf=[show_perform, sp_run_num],
                     outer=30, yscale="linear", ylim=[], ylabel="Step per episode",
                     right_ax=[],
                     label_ncol=2, true_perf_label=False)

if __name__ == '__main__':
    ranges = [0]
    # sweep_cem_uniform_baseline()
    # sweep_gridsearch_uniform_baseline()
    top_param()