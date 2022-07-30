pr_rnd = [34, 4, 43, 30, 24, 32, 40, 11, 20, 30, 3, 16, 53, 45, 0, 21, 43, 23, 44, 50, 9, 41, 37, 37, 11, 2, 26, 33, 18, 20]
pr_true = ["../../data/icml_data/puddlerand/online_learning/esarsa/step30k/sweep/"]

pr_knnlaplace_optim_5k = ["../../data/icml_data/puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step5k_env/data_optimal/drop0/sweep_rep1/"]
pr_knnlaplace_suboptim_500 = ["../../data/hyperparam_v5/puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step500_env/data_suboptimal/drop0/sweep_rep1/"]

pr_cem_uniform = ["../../data/hyperparam_v5/puddlerand/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step500_env/data_suboptimal/drop0/cem_rand_sample"]
# pr_cem_uniform_best = ["../../data/hyperparam_v5/puddlerand/online_learning/esarsa/step30k/cem_uniform_baseline_best"]
pr_cem_uniform_online = ["../../data/hyperparam_v5/puddlerand/online_learning/esarsa/step30k/cem_uniform_sample"]

pr_bayes_online = ["../../data/hyperparam_v5/puddlerand/online_learning/esarsa/step30k/bayesopt/"]
pr_randomsearch_online = ["../../data/hyperparam_v5/puddlerand/online_learning/esarsa/step30k/randomsearch/"]

pr_gridsearch_uniform = ["../../data/hyperparam_v5/puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step5k_env/data_optimal/drop0/gridsearch_uniform_sample"]
# pr_gridsearch_uniform_best = ["../../data/hyperparam_v5/puddlerand/online_learning/esarsa/step30k/gridsearch_uniform_baseline_best"]
pr_gridsearch_uniform_online = ["../../data/hyperparam_v5/puddlerand/online_learning/esarsa/step30k/gridsearch_uniform_sample"]

pr_cemlaplace = ["../../data/hyperparam_v5/puddlerand/online_learning/esarsa/step30k/CEMOnlineLearningListSweep_100iters"]
# pr_cemlaplace = ["../../data/icml_data/puddlerand/online_learning/esarsa/step30k/CEMOnlineLearningListSweep_30iters"]


pr_networkscaledlaplace_optim_5k = ["../../data/icml_data/puddlerand/offline_learning/network/learning/clip_scale_laplace_separated/timeout1000/esarsa/step5k_env/data_optimal/sweep/"]
pr_fqi_tc = ["../../data/icml_data/puddlerand/offline_learning/fqi-linear/fqi-adam/alpha_hidden_epsilon/step5k_env/optimalfixed_eps0/lambda1e-3/lockat_baseline_online/"]


pr_cql_offline_temp = "../../pylib/data/output/test_v0/puddlerand/cql_offline/data5k_eps0/sweep_{}/"
pr_cql_online_temp = "../../data/hyperparam_v5/puddlerand/online_learning/cql/step30k/sweep_{}/"
pr_cql_offline = []
pr_cql_online = []
for i in range(30):
    pr_cql_offline.append(pr_cql_offline_temp.format(i))
    pr_cql_online.append(pr_cql_online_temp.format(i))
