ac_rnd = [34, 4, 43, 30, 24, 32, 40, 11, 20, 30, 3, 16, 53, 45, 0, 21, 43, 23, 44, 50, 9, 41, 37, 37, 11, 2, 26, 33, 18, 20]
ac_true = ["../../data/icml_data/acrobot/online_learning/esarsa/step15k/sweep/"] # used

ac_knnlaplace_optim_5k = ["../../data/icml_data/acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step5k_env/data_optimal/drop0/sweep_rep1/"]
ac_knnlaplace_suboptim_500 = ["../../data/icml_data/acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step500_env/data_suboptimal/drop0/sweep_rep1/"]

ac_cem_uniform = ["../../data/hyperparam_v5/acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step500_env/data_suboptimal/drop0/cem_rand_sample"]
# ac_cem_uniform_best = ["../../data/hyperparam_v5/acrobot/online_learning/esarsa/step15k/cem_uniform_baseline_best"]
ac_cem_uniform_online = ["../../data/hyperparam_v5/acrobot/online_learning/esarsa/step15k/cem_uniform_sample"]

ac_bayes_online = ["../../data/hyperparam_v5/acrobot/online_learning/esarsa/step15k/bayesopt/"]
ac_randomsearch_online = ["../../data/hyperparam_v5/acrobot/online_learning/esarsa/step15k/randomsearch/"]

# ac_gridsearch_uniform = ["../../data/hyperparam_v5/acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step5k_env/data_optimal/drop0/gridsearch_uniform_sample/param_{}"]
ac_gridsearch_uniform = ["../../data/hyperparam_v5/acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step5k_env/data_optimal/drop0/gridsearch_uniform_sample"]
# ac_gridsearch_uniform_best = ["../../data/hyperparam_v5/acrobot/online_learning/esarsa/step15k/gridsearch_uniform_baseline_best"]
ac_gridsearch_uniform_online = ["../../data/hyperparam_v5/acrobot/online_learning/esarsa/step15k/gridsearch_uniform_sample"]

ac_cemlaplace = ["../../data/icml_data/acrobot/online_learning/esarsa/step15k/CEMOnlineLearningListSweep_100iters"]

ac_networkscaledlaplace_optim_5k = ["../../data/icml_data/acrobot/offline_learning/network/learning/clip_scale_laplace_separated/timeout500/esarsa/step5k_env/data_optimal/sweep/"]
ac_fqi_tc = ["../../data/icml_data/acrobot/offline_learning/fqi-linear/fqi-adam/alpha_hidden_epsilon/step5k_env/optimalfixed_eps0/lambda1e-3/lockat_baseline_online/"]

ac_cql_offline_temp = "../../pylib/data/output/test_v0/acrobot/cql_offline/data5k_eps0/sweep_{}/"
ac_cql_online_temp = "../../data/hyperparam_v5/acrobot/online_learning/cql/step15k/sweep_{}/"
ac_cql_offline = []
ac_cql_online = []
for i in range(30):
    ac_cql_offline.append(ac_cql_offline_temp.format(i))
    ac_cql_online.append(ac_cql_online_temp.format(i))
