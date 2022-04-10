import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box2.paths_acrobot_shift import *

def sweep_true_perform():
    calibration = {
        "Knn": ac_knnlaplace_optim_5k_pi,
    }
    true = {"true": ac_true_pi}

    plot_compare_top(true, calibration, None, [], "totals", "../img/temp",
                     outer=30, res_scale=-1, yscale="linear", ylim=[], ylabel="Step per episode",
                     label_ncol=2, true_perf_label=False)

def learning_curve():
    path = ac_actorcritic_knnlaplace_optim[0]
    print("Load result from", path)
    params = list(range(16)) #[18] #[18, 19, 27, 28] #
    plot_learning_curve(path, params, mode="returns", num_runs=2, single_run=False, smooth=1)

def data_distribution():
    paths = {
        "no pi init": ac_true,
        "true": ac_esarsa_true_trans,
        # "calibration": ac_pitrans_calibration_learning15k
    }
    params = {
        "no pi init": [18, 27],
        "true": [18, 19, 27, 28],
        # "calibration": [18, 27],
    }
    plot_perf_distribution(paths, params, "totals", flip=True, save_path="../img/temp")


def flip_top_param():
    calibration = {
        "w/o policy": ac_knnlaplace_optim_5k,
    }
    random = ac_rnd
    fqi = {"FQI": acflip_fqi_tc}
    true = {"true": acflip_true}

    show_perform = {
        "Init policy (learning 15k)": acflip_pitrans_calibration_learning15k,
        "Shift Esarsa transfer (true)": acflip_esarsa_true_trans,
    }
    sp_run_num = {
        "Init policy (learning 15k)": ac_knnlaplace_optim_5k,
        "Shift Esarsa transfer (true)": ac_true,
    }

    plot_compare_top(true, calibration, fqi, random, "totals", "../img/acflip_15k",
                     load_perf=[show_perform, sp_run_num],
                     outer=30, res_scale=-1, yscale="linear", ylim=[], ylabel="Step per episode",
                     right_ax=["Random", "FQI", "Shift Esarsa transfer (calibration)"],
                     label_ncol=2, true_perf_label=False)

def shift_top_param():
    calibration = {
        "w/o policy": ac_knnlaplace_optim_5k,
    }
    random = ac_rnd
    fqi = {"FQI": acshift_fqi_tc}
    true = {"true": acshift_true}

    show_perform = {
        "Init policy (learning 15k)": acshift_pitrans_calibration_learning15k,
        "Shift Esarsa transfer (true)": acshift_esarsa_true_trans,
    }
    sp_run_num = {
        "Init policy (learning 15k)": ac_knnlaplace_optim_5k,
        "Shift Esarsa transfer (true)": ac_true,
    }

    plot_compare_top(true, calibration, fqi, random, "totals", "../img/acshift_15k",
                     load_perf=[show_perform, sp_run_num],
                     outer=30, res_scale=-1, yscale="linear", ylim=[], ylabel="Step per episode",
                     right_ax=["Random", "FQI", "Shift Esarsa transfer (calibration)"],
                     label_ncol=2, true_perf_label=False)

    # show_perform = {
    #     "Init policy (learning 50k)": acshift_pitrans_calibration_learning50k,
    #     "Shift Esarsa transfer (true)": acshift_esarsa_true_trans,
    # }
    # true = {"true": acshift_true_long}
    # plot_compare_top(true, calibration, fqi, random, "totals", "../img/acshift_50k",
    #                  load_perf=show_perform,
    #                  outer=30, res_scale=-1, yscale="linear", ylim=[240, 320], ylabel="Step per episode",
    #                  right_ax=["Random", "FQI", "Shift Esarsa transfer (true)", "Shift Esarsa transfer (calibration)"],
    #                  label_ncol=2, true_perf_label=False)

def default_top_param():
    calibration = {
        "w/o policy": ac_knnlaplace_optim_5k,
    }
    random = ac_rnd
    fqi = {"FQI": ac_fqi_tc}
    true = {"true": ac_true}

    show_perform = {
        "Init policy (learning 15k)": ac_pitrans_calibration_learning15k,
        # "init policy (learning 50k)": ac_pitrans_calibration_learning50k,
        "Shift Esarsa transfer (true)": ac_esarsa_true_trans,
        # "Sanity Check lock weight": ac_pitrans_true_lock_weight,
        # "Sanity Check lr=0": ac_pitrans_true_lr0,
    }
    sp_run_num = {
        "Init policy (learning 15k)": ac_knnlaplace_optim_5k,
        "Shift Esarsa transfer (true)": ac_true,
        # "Sanity Check lock weight": ac_true_long,
        # "Sanity Check lr=0": ac_true_long,
    }
    plot_compare_top(true, calibration, fqi, random, "totals", "../img/acdefault_15k",
                     load_perf=[show_perform, sp_run_num],
                     # outer=30, res_scale=-1, yscale="linear", ylim=[110, 210], ylabel="Step per episode",
                     outer=30, res_scale=-1, yscale="linear", ylim=[], ylabel="Step per episode",
                     right_ax=["Random", "FQI"],
                     label_ncol=2, true_perf_label=False)

    # show_perform = {
    #     "Init policy (learning 50k)": ac_pitrans_calibration_learning50k,
    # }
    # true = {"true": ac_true_long}
    # plot_compare_top(true, calibration, fqi, random, "totals", "../img/acdefault_50k",
    #                  load_perf=show_perform,
    #                  outer=30, res_scale=-1, yscale="linear", ylim=[100, 116], ylabel="Step per episode",
    #                  right_ax=["Random", "FQI"],
    #                  label_ncol=2, true_perf_label=False)

def default_top_param_piinit():
    calibration = {
        "Knn": ac_knnlaplace_optim_5k_pi,
    }
    random = ac_rnd
    fqi = {"FQI": ac_fqi_tc}
    true = {"true": ac_true_pi}
    plot_compare_top(true, calibration, fqi, random, "totals", "../img/ac_15k_piinit",
                     outer=30, res_scale=-1, yscale="linear", ylim=[89, 250.5], ylabel="Step per episode",
                     right_ax=["Random", "FQI", "Shift Esarsa transfer (calibration)"],
                     label_ncol=2, true_perf_label=False)
def shift_top_param_piinit():
    calibration = {
        "Knn": ac_knnlaplace_optim_5k_pi,
    }
    random = ac_rnd
    fqi = {"FQI": acshift_fqi_tc}
    true = {"true": acshift_true_pi}
    plot_compare_top(true, calibration, fqi, random, "totals", "../img/acshift_15k_piinit",
                     outer=30, res_scale=-1, yscale="linear", ylim=[160, 950], ylabel="Step per episode",
                     right_ax=["Random", "FQI", "Shift Esarsa transfer (calibration)"],
                     label_ncol=2, true_perf_label=False)
def flip_top_param_piinit():
    calibration = {
        "Knn": ac_knnlaplace_optim_5k_pi,
    }
    random = ac_rnd
    fqi = {"FQI": acflip_fqi_tc}
    true = {"true": acflip_true_pi}
    plot_compare_top(true, calibration, fqi, random, "totals", "../img/acflip_15k_piinit",
                     outer=30, res_scale=1, yscale="linear", ylim=[], ylabel="Step per episode",
                     # right_ax=["Random", "FQI", "Shift Esarsa transfer (calibration)"],
                     label_ncol=2, true_perf_label=False)

if __name__ == '__main__':
    ranges = [0]
    # sweep_true_perform()
    # learning_curve()
    # data_distribution()
    # flip_top_param()
    # shift_top_param()
    # default_top_param()
    # flip_top_param_piinit()
    # shift_top_param_piinit()
    # default_top_param_piinit()
