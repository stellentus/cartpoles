cart_rnd = [34, 4, 43, 30, 24, 32, 40, 11, 20, 30, 3, 16, 53, 45, 0, 21, 43, 23, 44, 50, 9, 41, 37, 37, 11, 2, 26, 33, 18, 20]
basepath = "../../data/icml_data/"

cart_true = [basepath + "cartpole/online_learning/esarsa/step50k/sweep/"] # used
cart_knnlaplace_optim_10k_plot7 = [basepath + "cartpole/offline_learning/knn/learning/k3_laplace/timeout1666/esarsa/step10k_env/data_optimal/drop0/sweep_rep1/"] # used
cart_knnlaplace_suboptim_10k_plot7 = [basepath + "cartpole/offline_learning/knn/learning/k3_laplace/timeout1666/esarsa/step10k_env/data_suboptimal/drop0/sweep_rep1/"] # used
cart_knnlaplace_random_10k_plot7 = [basepath + "cartpole/offline_learning/knn/learning/k3_laplace/timeout1666/esarsa/step10k_env/data_random/drop0/sweep_rep1/"] # used
cart_knnlaplace_learningpolicy_10k_plot7 = [basepath + "cartpole/offline_learning/knn/learning/k3_laplace/timeout1666/esarsa/step10k_env/data_learningpolicy/drop0/sweep_rep1/"]

cart_knn_combined_15k_plot7 = ["../../data/hyperparam_v5/cartpole/offline_learning/knn/learning/k3/timeout1666/esarsa/step15k_env/data_mixed/drop0/sweep"]
cart_knn_suboptrand_15k_drop20 = ["../../data/hyperparam_v5/cartpole/offline_learning/knn/learning/k3/timeout1666/esarsa/step10k_env/data_subopt_random/drop20/sweep"]
cart_knnlaplace_random_10k_drop20 = ["../../data/hyperparam_v5/cartpole/offline_learning/knn/learning/k3_laplace/timeout1666/esarsa/step10k_env/data_random/drop20/sweep_rep1"]