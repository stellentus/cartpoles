ac_rnd = [34, 4, 43, 30, 24, 32, 40, 11, 20, 30, 3, 16, 53, 45, 0, 21, 43, 23, 44, 50, 9, 41, 37, 37, 11, 2, 26, 33, 18, 20]
basepath = "../../data/icml_data/"
basepath_new = "../../data/icml_data/"

ac_true = [basepath + "acrobot/online_learning/esarsa/step15k/sweep/"] # used
ac_fqi_tc = [basepath + "acrobot/offline_learning/fqi-linear/fqi-adam/alpha_hidden_epsilon/step5k_env/optimalfixed_eps0/lambda1e-3/lockat_baseline_online/"] # used
ac_fqi_nn = [basepath + "acrobot/offline_learning/fqi/fqi-adam/alpha_hidden_epsilon/step5k_env/optimalfixed_eps0/earlystop/lambda1e-3/lockat_baseline_online/"] # used


# PLOT 1
ac_knnlaplace_optim_5k = [basepath + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step5k_env/data_optimal/drop0/sweep_rep1/"] # used
ac_knnraw_optim_5k = [basepath + "acrobot/offline_learning/knn/learning/k3/timeout500/esarsa/step5k_env/data_optimal/drop0/sweep/"] # used
ac_networkscaledlaplace_optim_5k = [basepath + "acrobot/offline_learning/network/learning/clip_scale_laplace_separated/timeout500/esarsa/step5k_env/data_optimal/sweep/"] # used
ac_networkscaledraw_optim_5k = [basepath + "acrobot/offline_learning/network/learning/clip_scale_separated/timeout500/esarsa/step5k_env/data_optimal/sweep/"] #used

# PLOT 2
ac_knnlaplace_avg_5k_new = [basepath_new + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step5k_env/data_suboptimal/drop0/sweep_rep1/"] # used
ac_knnlaplace_avg_2500_new = [basepath_new + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step2.5k_env/data_suboptimal/drop0/sweep_rep1/"]
ac_knnlaplace_avg_1k_new = [basepath_new + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step1k_env/data_suboptimal/drop0/sweep_rep1/"] # used
ac_knnlaplace_avg_500_new = [basepath_new + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step500_env/data_suboptimal/drop0/sweep_rep1/"] # used

ac_knnlaplace_optim_5k_new = [basepath_new + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step5k_env/data_optimal/drop0/sweep_rep1/"] # used
ac_knnlaplace_bad_5k_new = [basepath_new + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step5k_env/data_subsuboptimal/drop0/sweep_rep1/"] # used

# PLOT 3
acshift_true = [basepath_new + "acrobot_shift/online_learning/shift/esarsa/step15k/sweep/"] # used
acshift_knnlaplace_optim_5k = [basepath + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step5k_env/data_optimal/drop0/sweep_rep1/"] # used
acshift_esarsa_true_trans = [basepath + "acrobot_shift/policy_transfer/shift/load_default/esarsa/all/"] # used
# acshift_esarsa_calibration_trans = ["../../data/hyperparam_v5/acrobot_shift/policy_transfer/shift/load_calibration_default/esarsa/fixed_all/"] # used
acshift_esarsa_calibration_trans = [basepath + "acrobot_shift/policy_transfer/shift/load_calibration_default/esarsa/fixed_all/"] # used
acshift_fqi_tc_optim_5k = [basepath + "acrobot_shift/policy_transfer/shift/load_default/fqi-linear/fqi-adam/alpha_hidden_epsilon/step5k_env/optimalfixed_eps0/lambda1e-3/lockat_baseline_online/"]
acshift_fqi_nn_optim_5k = [basepath + "acrobot_shift/policy_transfer/shift/load_default/fqi/fqi-adam/alpha_hidden_epsilon/step5k_env/optimalfixed_eps0/earlystop/lambda1e-3/lockat_baseline_online"]
# acshift_fqi_tc_optim_5k = [basepath + "acrobot_shift/policy_transfer/shift/load_default/fqi-linear/lambda1e-3/"]
# acshift_fqi_nn_optim_5k = [basepath + "acrobot_shift/policy_transfer/shift/load_default/fqi/lambda1e-3/"]

# PLOT temp
ac_dqn = [basepath + "acrobot/online_learning/dqn/step600k/sweep/"]
ac_actorcritic = [basepath + "acrobot/online_learning/ac/step30k/sweep/"] # used
ac_dqn_knnlaplace_optim = [basepath + "acrobot/offline_learning/knn/learning/k3_laplace/timeout2k/dqn/step5k_env/data_optimal/drop0/sweep_rep1/"]
ac_actorcritic_knnlaplace_optim = [basepath + "acrobot/offline_learning/knn/learning/k3_laplace/timeout500/ac/step5k_env/data_optimal/drop0/sweep_rep1/"] # used

# PLOT CEM
ac_cemlaplace_optim_5k = [basepath + "acrobot/online_learning/esarsa/step15k/CEMOnlineLearningListSweep_100iters"]

ac_true_old = [basepath + "hyperparam_ap_CEM_gridsearch/data/hyperparam_ap/acrobot/online_learning/esarsa/step15k/sweep/"]
ac_knnraw_optim_5k_old = [basepath + "hyperparam_ap_CEM_gridsearch/data/hyperparam_ap/acrobot/offline_learning/k3_timeout750/esarsa/step15k/optimalfixed_eps0/sweep/"]
ac_cemraw_optim_5k_old = [basepath + "hyperparam_ap_CEM_gridsearch/data/hyperparam_ap/acrobot/list/CEMoffline_onlineEvaluation/esarsa/step15k/sweep/"]
