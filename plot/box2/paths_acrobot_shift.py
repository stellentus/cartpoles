# ac_knnlaplace_optim_5k = ["../../data/hyperparam_v5/acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step5k_env/data_optimal/drop0/sweep_rep1/"]
ac_knnlaplace_optim_5k = ["../../data/hyperparam_v5/acrobot_shift/calibration_learning/default/k3_laplace/timeout500/esarsa/step5k_env/data_optimal/drop0/step15k_all/"]
ac_knnlaplace_optim_5k_pi = ["../../data/hyperparam_v5/acrobot_shift/calibration_learning/default/k3_laplace/timeout500/esarsa/step5k_env/data_optimal/drop0/step15k_pi_sweep/"]


ac_rnd = [34, 4, 43, 30, 24, 32, 40, 11, 20, 30, 3, 16, 53, 45, 0, 21, 43, 23, 44, 50, 9, 41, 37, 37, 11, 2, 26, 33, 18, 20]


ac_true = ["../../data/hyperparam_v5/acrobot/online_learning/esarsa/step15k/sweep/"]
ac_true_pi = ["../../data/hyperparam_v5/acrobot_shift/online_learning/default/esarsa/step15k/pi_sweep/"]

acshift_true = ["../../data/hyperparam_v5/acrobot_shift/online_learning/shift/esarsa/step15k/sweep/"]
acshift_true_pi = ["../../data/hyperparam_v5/acrobot_shift/online_learning/shift/esarsa/step15k/pi_sweep/"]

acflip_true = ["../../data/hyperparam_v5/acrobot_shift/online_learning/act_flip/esarsa/step15k/sweep/"]
acflip_true_pi = ["../../data/hyperparam_v5/acrobot_shift/online_learning/act_flip/esarsa/step15k/pi_sweep/"]


ac_fqi_tc = ["../../data/icml_data/acrobot/offline_learning/fqi-linear/fqi-adam/alpha_hidden_epsilon/step5k_env/optimalfixed_eps0/lambda1e-3/lockat_baseline_online/"]
ac_fqi_nn = ["../../data/icml_data/acrobot/offline_learning/fqi/fqi-adam/alpha_hidden_epsilon/step5k_env/optimalfixed_eps0/earlystop/lambda1e-3/lockat_baseline_online/"]

acshift_fqi_tc = ["../../data/hyperparam_v7/acrobot_shift/offline_learning/random_restarts/fqi-linear/fqi-adam/alpha_hidden_epsilon/step5k_env/optimalfixed_eps0/lambda1e-3/lockat_baseline_online/"]
acshift_fqi_nn = ["../../data/hyperparam_v7/acrobot_shift/offline_learning/random_restarts/fqi/fqi-adam/alpha_hidden_epsilon/step5k_env/optimalfixed_eps0/lambda1e-3/lockat_baseline_online/"]

acflip_fqi_tc = ["../../data/hyperparam_v7/acrobot_act_flip/offline_learning/random_restarts/fqi-linear/fqi-adam/alpha_hidden_epsilon/step5k_env/optimalfixed_eps0/lambda1e-3/lockat_baseline_online/"]
acflip_fqi_nn = ["../../data/hyperparam_v7/acrobot_act_flip/offline_learning/random_restarts/fqi/fqi-adam/alpha_hidden_epsilon/step5k_env/optimalfixed_eps0/lambda1e-3/lockat_baseline_online/"]


ac_pitrans_calibration_learning15k = ["../../data/hyperparam_v5/acrobot_shift/policy_transfer/default/load_calibration_default/esarsa/learning/"]
acflip_pitrans_calibration_learning15k = ["../../data/hyperparam_v5/acrobot_shift/policy_transfer/act_flip/load_calibration_default/esarsa/learning/"]
acshift_pitrans_calibration_learning15k = ["../../data/hyperparam_v5/acrobot_shift/policy_transfer/shift/load_calibration_default/esarsa/learning/"]

# ac_pitrans_calibration_learning15k = ["../../data/hyperparam_v5/acrobot_shift/policy_transfer/default/load_calibration_default/esarsa/learning_init_pi16/"]
# acflip_pitrans_calibration_learning15k = ["../../data/hyperparam_v5/acrobot_shift/policy_transfer/act_flip/load_calibration_default/esarsa/learning_init_pi16/"]
# acshift_pitrans_calibration_learning15k = ["../../data/hyperparam_v5/acrobot_shift/policy_transfer/shift/load_calibration_default/esarsa/learning_init_pi16/"]


ac_esarsa_true_trans = ["../../data/hyperparam_v5/acrobot_shift/policy_transfer/default/load_default/esarsa/learning/"]
acflip_esarsa_true_trans = ["../../data/hyperparam_v5/acrobot_shift/policy_transfer/act_flip/load_default/esarsa/learning/"]
# acshift_esarsa_true_trans = ["../../data/hyperparam_v5/acrobot_shift/policy_transfer/shift/load_default/esarsa/learning/"] # learning
acshift_esarsa_true_trans = ["../../data/icml_data/acrobot_shift/policy_transfer/shift/load_default/esarsa/all"] # fixed

# ac_esarsa_true_trans = ["../../data/hyperparam_v5/acrobot_shift/policy_transfer/default/load_default/esarsa/learning_init_pi16/"]
# acflip_esarsa_true_trans = ["../../data/hyperparam_v5/acrobot_shift/policy_transfer/act_flip/load_default/esarsa/learning_init_pi16/"]
# acshift_esarsa_true_trans = ["../../data/hyperparam_v5/acrobot_shift/policy_transfer/shift/load_default/esarsa/learning_init_pi16/"]


# ac_pitrans_true_lock_weight = ["../../data/hyperparam_v5/acrobot_shift/policy_transfer/default/sanity_check/load_default/lock_weight/"]
# ac_pitrans_true_lr0 = ["../../data/hyperparam_v5/acrobot_shift/policy_transfer/default/sanity_check/load_default/lr0/"]

ac_actorcritic_knnlaplace_optim = ["../../data/icml_data/acrobot/offline_learning/knn/learning/k3_laplace/timeout500/ac/step5k_env/data_optimal/drop0/sweep_rep1/"]

acshift_knnlaplace_optim_5k = ["../../data/icml_data/acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step5k_env/data_optimal/drop0/sweep_rep1/"] # used
acshift_esarsa_calibration_trans = ["../../data/icml_data/acrobot_shift/policy_transfer/shift/load_calibration_default/esarsa/fixed_all/"] # used
acshift_fqi_tc_optim_5k = ["../../data/icml_data/acrobot_shift/policy_transfer/shift/load_default/fqi-linear/fqi-adam/alpha_hidden_epsilon/step5k_env/optimalfixed_eps0/lambda1e-3/lockat_baseline_online/"]

ac_cql_offline_temp = "../../pylib/data/output/test_v0/acrobot/cql_offline/data5k_eps0/sweep_{}/"
acshift_cql_online_temp = "../../data/hyperparam_v5/acrobot_shift/online_learning/cql/step15k/sweep_{}/"
ac_cql_offline = []
acshift_cql_online = []
for i in range(30):
    ac_cql_offline.append(ac_cql_offline_temp.format(i))
    acshift_cql_online.append(acshift_cql_online_temp.format(i))
