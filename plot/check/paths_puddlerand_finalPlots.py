pr_rnd = [34, 4, 43, 30, 24, 32, 40, 11, 20, 30, 3, 16, 53, 45, 0, 21, 43, 23, 44, 50, 9, 41, 37, 37, 11, 2, 26, 33, 18, 20]
pr_rnd_30 = [16, 28, 19, 6, 18, 26, 28, 11, 14, 0, 9, 22, 11, 15, 18, 15, 1, 11, 20, 20, 9, 17, 1, 1, 11, 20, 2, 3, 18, 26]
basepath = "../../data/icml_data/"
basepath_new = "../../data/icml_data/"
basepath_cem = "../../data/icml_data/"

pr_true = [basepath + "puddlerand/online_learning/esarsa/step30k/sweep/"] # used
pr_fqi_tc = [basepath + "puddlerand/offline_learning/fqi-linear/fqi-adam/alpha_hidden_epsilon/step5k_env/optimalfixed_eps0/lambda1e-3/lockat_baseline_online/"] # used
pr_fqi_nn = [basepath + "puddlerand/offline_learning/fqi/fqi-adam/alpha_hidden_epsilon/step5k_env/optimalfixed_eps0/earlystop/lambda1e-3/lockat_baseline_online/"] # used

# PLOT 1
pr_knnlaplace_optim_5k = [basepath + "puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step5k_env/data_optimal/drop0/sweep_rep1/"] # used
pr_knnraw_optim_5k = [basepath + "puddlerand/offline_learning/knn/learning/k3/timeout1000/esarsa/step5k_env/data_optimal/drop0/sweep/"] # used
pr_networkscaledlaplace_optim_5k = [basepath + "puddlerand/offline_learning/network/learning/clip_scale_laplace_separated/timeout1000/esarsa/step5k_env/data_optimal/sweep/"] # used
pr_networkscaledraw_optim_5k = [basepath + "puddlerand/offline_learning/network/learning/clip_scale_separated/timeout1000/esarsa/step5k_env/data_optimal/sweep/"] # used

# PLOT 2
pr_knnlaplace_avg_5k_new = [basepath_new + "puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step5k_env/data_suboptimal/drop0/sweep_rep1/"] # used
pr_knnlaplace_avg_2500_new = [basepath_new + "puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step2.5k_env/data_suboptimal/drop0/sweep_rep1/"]
pr_knnlaplace_avg_1k_new = [basepath_new + "puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step1k_env/data_suboptimal/drop0/sweep_rep1/"] # used
pr_knnlaplace_avg_500_new = [basepath_new + "puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step500_env/data_suboptimal/drop0/sweep_rep1/"] # used

pr_knnlaplace_optim_5k_new = [basepath_new + "puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step5k_env/data_optimal/drop0/sweep_rep1/"] # used
pr_knnlaplace_bad_5k_new = [basepath_new + "puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step5k_env/data_subsuboptimal/drop0/sweep_rep1/"] # used

# PLOT 4

# pr_knnlaplace_optim_5k_plot3 = [basepath + "puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step5k_env/data_optimal/drop0/sweep_rep1/"]
# pr_knnlaplace_suboptim_5k_plot3 = [basepath + "puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step5k_env/data_suboptimal/drop0/sweep_rep1/"]
# pr_knnlaplace_subsuboptim_5k_plot3 = [basepath + "puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step5k_env/data_subsuboptimal/drop0/sweep_rep1/"]

# PLOT agent
pr_dqn = [basepath + "puddlerand/online_learning/dqn/step600k/sweep/"]
pr_actorcritic = [basepath + "puddlerand/online_learning/ac/step30k/sweep/"]
pr_dqn_knnlaplace_optim = [basepath + "puddlerand/offline_learning/knn/learning/k3_laplace/timeout2k/dqn/step5k_env/data_optimal/drop0/sweep_rep1/"]
# pr_actorcritic_knnlaplace_optim = [basepath + "puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/ac/step15k_env/data_optimal/drop0/sweep_rep1/"]
pr_actorcritic_knnlaplace_optim = [basepath + "puddlerand/offline_learning/knn/learning/k3_laplace/timeout500/ac/step30k_env/data_optimal/drop0/sweep_rep0/"]

# PLOT CEM
pr_cemlaplace_optim_5k = [basepath + "puddlerand/list/CEMofflineList_KNNlaplace/esarsa/step30k/sweep/"] # used

#pr_true_old = [basepath + "hyperparam_ap_CEM_gridsearch/data/hyperparam_ap/puddleworld/online_learning/esarsa/step30k/sweep/"]
#pr_knnraw_optim_5k_old = [basepath + "hyperparam_ap_CEM_gridsearch/data/hyperparam_ap/puddleworld/offline_learning/k3_timeout400/esarsa/step30k/optimalfixed_eps0/sweep/"]
#pr_cemraw_optim_5k_old = [basepath + "hyperparam_ap_CEM_gridsearch/data/hyperparam_ap/puddleworld/list/CEMoffline_onlineEvaluation/esarsa/step30k/sweep/"]

 
pr_true_cem = [basepath_cem + "onlineLearning/data/puddleworld/sweep/"] # used

pr_k3_suboptim_500data = [basepath_cem + "offlineLearning/puddleKNNresults/data/hyperparam_v5/puddlerand/offline_learning/knn/learning/k3/timeout500/esarsa/step500_env/data_suboptimal/drop0/sweep/"]
pr_k3_laplace_suboptim_500data = [basepath_cem + "offlineLearning/puddleKNNresults/data/hyperparam_v5/puddlerand/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step500_env/data_suboptimal/drop0/sweep_rep1/"] # used

pr_CEM_k3_suboptim_500data_50iters = [basepath_cem + "onlineLearning/data/hyperparam_ap/puddleworld/list/CEMofflineList/esarsa/step15k/cem_k3_online/sweep_50iters/"]
pr_CEM_k3_laplace_suboptim_500data_50iters = [basepath_cem + "onlineLearning/data/hyperparam_ap/puddleworld/list/CEMofflineList/esarsa/step15k/cem_k3_laplace_online/sweep_50iters/"]

pr_CEM_k3_suboptim_500data_100iters = [basepath_cem + "onlineLearning/data/hyperparam_ap/puddleworld/list/CEMofflineList/esarsa/step15k/cem_k3_online/sweep_100iters/"]
# pr_CEM_k3_laplace_suboptim_500data_100iters = [basepath_cem + "onlineLearning/data/hyperparam_ap/puddleworld/list/CEMofflineList/esarsa/step15k/cem_k3_laplace_online/sweep_100iters/"] # used
pr_CEM_k3_laplace_suboptim_500data_100iters = [basepath_cem + "puddleworld/online_learning/CEMofflineList/esarsa/step15k/cem_k3_laplace_online/sweep_100iters/"] # used
