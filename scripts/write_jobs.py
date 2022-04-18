def write_grid_search(sweep, run, config_list, prev=0, line_per_file=1, sweep_start=0, run_start=0, sweep_list=[]):
    f = open("tasks_{}.sh".format(prev), "w")
    if sweep_list == []:
        sweep_list = list(range(sweep_start, sweep))
    count = 0
    for config in config_list:
        for r in range(run_start, run):
            for s in sweep_list:
                f.write("./main -config {} -run {} -sweep {}\n".format(config, r, s))
                # f.write("go run cmd/experiment/main.go -config {} -run {} -sweep {}\n".format(config, r, s))
                count += 1
                if count % line_per_file == 0:
                    f.close()
                    prev += 1
                    f = open("tasks_{}.sh".format(prev), "w")

def write_cem(fi, parallel=20, hours=8):
    f = open("run_node_{}.sh".format(fi), "w")
    count = 0
    f.writelines(
        ["#!/bin/bash \n",
         "#SBATCH --account=rrg-whitem\n",
         "#SBATCH --mail-type=ALL\n",
         "#SBATCH --mail-user=han8@ualberta.ca\n",
         "#SBATCH --nodes=1\n",
         "#SBATCH --ntasks={}\n".format(parallel),
         "#SBATCH --cpus-per-task=1\n",
         "#SBATCH --time={}:55:00\n".format(hours-1),
         "#SBATCH --mem-per-cpu=4G\n",
         "#SBATCH --job-name cem{}\n".format(fi),
         "#SBATCH --output=out.txt\n",
         "#SBATCH --error=err.txt\n",

         "chmod +x tasks*\n",
         "cd $SLURM_SUBMIT_DIR/../\n",
         "chmod +x main\n",
         "export OMP_NUM_THREADS=1\n",
         "source $HOME/gpu_env/bin/activate\n",
         "./main -datasetSeed={} > dataset{}_logs.txt \n".format(fi, fi),
         "sleep {}h\n".format(hours)]
    )
# def write_cem(run, prev=0, line_per_file=1):
#     f = open("tasks_{}.sh".format(prev), "w")
#     count = 0
#     for r in range(run):
#
#
#         f.write("./main -datasetSeed={} > dataset{}_logs.txt \n".format(r, r))
#         # f.write("go run cmd/experiment/main.go -config {} -run {} -sweep {}\n".format(config, r, s))
#         count += 1
#         if count % line_per_file == 0:
#             f.close()
#             prev += 1
#             f = open("tasks_{}.sh".format(prev), "w")

def write_bayes_opt(configs, start_script, num_cpu, num_inner_run, num_data=30):
    start = start_script
    for config in configs:
        for i in range(start, start+num_data):
            f = open("tasks_{}.sh".format(i), "w")
            f.write("python bayesopt.py --num-runs {} --config-file {} --max_cpu {} --log-idx {}".format(num_inner_run, config, num_cpu, i-start))
            f.close()
        start += num_data
        print("Done config {}, end script {}".format(config, start))

import math
def write_script(start_script, num_script, start_task, total_tasks, hours, min_node, parallel):
    task_per_script = math.ceil((total_tasks-start_task+1) / float(num_script))
    count = start_task
    fi = start_script
    while count < total_tasks:
        f = open("run_node_{}.sh".format(fi), "w")
        f.writelines(
            ["#!/bin/bash \n",
             "#SBATCH --account=rrg-whitem\n",
             "#SBATCH --mail-type=ALL\n",
             "#SBATCH --mail-user=han8@ualberta.ca\n",
             "#SBATCH --nodes={}\n".format(min_node),
             "#SBATCH --ntasks={}\n".format(parallel),
             "#SBATCH --cpus-per-task=1\n",
             "#SBATCH --time={}:55:00\n".format(hours-1),
             "#SBATCH --mem-per-cpu=4G\n",
             "#SBATCH --job-name hyperparam{}\n".format(fi),
             "#SBATCH --output=out.txt\n",
             "#SBATCH --error=err.txt\n",

             "chmod +x tasks*\n",
             "cd $SLURM_SUBMIT_DIR/../\n",
             "chmod +x main\n",
             "export OMP_NUM_THREADS=1\n",
             "source $HOME/gpu_env/bin/activate\n",
             "parallel --jobs "+str(parallel)+" --results ./scripts/outputs"+str(fi)+"/ ./scripts/tasks_{}.sh ::: $(seq "+str(int(count))+" "+str(int(min(count+task_per_script-1, total_tasks)))+") &\n",
             "sleep {}h\n".format(hours)])
        f.close()
        count += task_per_script
        fi += 1
    return

# paths = []
#
# template = "config/hyperparam_v5/acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step5k_env/data_optimal/drop0/gridsearch_uniform_sample/param_{}.json"
# for i in range(54):
#     paths.append(template.format(i))
# template = "config/hyperparam_v5/puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step5k_env/data_optimal/drop0/gridsearch_uniform_sample/param_{}.json"
# for i in range(54):
#     paths.append(template.format(i))
# template = "config/hyperparam_v5/acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step500_env/data_suboptimal/drop0/cem_uniform_sample/param_{}.json"
# for i in range(100):
#     paths.append(template.format(i))
# template = "config/hyperparam_v5/puddlerand/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step500_env/data_suboptimal/drop0/cem_uniform_sample/param_{}.json"
# for i in range(100):
#     paths.append(template.format(i))
# write_grid_search(1, 300, paths, prev=0, line_per_file=240)#, run_start=17, sweep_list=list(range(0, 17))+list(range(30, 54)))
#
# paths = []
#
# template = "config/hyperparam_v5/acrobot/online_learning/esarsa/step15k/gridsearch_uniform_sample/param_{}.json"
# for i in range(54):
#     paths.append(template.format(i))
# template = "config/hyperparam_v5/puddlerand/online_learning/esarsa/step30k/gridsearch_uniform_sample/param_{}.json"
# for i in range(54):
#     paths.append(template.format(i))
# template = "config/hyperparam_v5/acrobot/online_learning/esarsa/step15k/cem_uniform_sample/param_{}.json"
# for i in range(100):
#     paths.append(template.format(i))
# template = "config/hyperparam_v5/puddlerand/online_learning/esarsa/step30k/cem_uniform_sample/param_{}.json"
# for i in range(100):
#     paths.append(template.format(i))
# write_grid_search(1, 30, paths, prev=385, line_per_file=120)#, run_start=17, sweep_list=list(range(0, 17))+list(range(30, 54)))

# template = "config/hyperparam_v5/puddlerand/online_learning/esarsa/step30k/cem_100iter/param_{}.json"
# for i in range(30):
#     paths.append(template.format(i))
# write_grid_search(1, 30, paths, prev=0, line_per_file=50)#, run_start=17, sweep_list=list(range(0, 17))+list(range(30, 54)))


# write_grid_search(1, 30, [
#     "config/hyperparam_v5/acrobot/online_learning/esarsa/step15k/gridsearch_uniform_baseline_best.json",
#     "config/hyperparam_v5/puddlerand/online_learning/esarsa/step30k/gridsearch_uniform_baseline_best.json",
# ], prev=0, line_per_file=1)#, run_start=17, sweep_list=list(range(0, 17))+list(range(30, 54)))

# write_grid_search(1, 30, [
#     "config/hyperparam_v7/acrobot/offline_data/random_restarts/esarsa/step30k/suboptimalfixed_eps0.json",
#     "config/hyperparam_v7/acrobot/offline_data/random_restarts/esarsa/step30k/subsuboptimalfixed_eps0.json",
#     "config/hyperparam_v7/acrobot/offline_data/true_restarts/esarsa/step30k/suboptimalfixed_eps0.json",
#     "config/hyperparam_v7/acrobot/offline_data/true_restarts/esarsa/step30k/subsuboptimalfixed_eps0.json",
#
#     "config/hyperparam_v7/puddlerand/offline_data/random_restarts/esarsa/step30k/suboptimalfixed_eps0.json",
#     "config/hyperparam_v7/puddlerand/offline_data/random_restarts/esarsa/step30k/subsuboptimalfixed_eps0.json",
# ] , prev=0, line_per_file=1) # 20 min / 3 per hour

# write_grid_search(150, 1, [
#     "config/hyperparam_v7/acrobot/offline_learning/fqi/fqi-adam/alpha_hidden_epsilon/step5k/eps0/weight_lambda1e-3.json",
#     "config/hyperparam_v7/acrobot/offline_learning/fqi-linear/fqi-adam/step5k/eps0/weight_lambda1e-3.json",
#
#     "config/hyperparam_v7/puddlerand/offline_learning/fqi/fqi-adam/alpha_hidden_epsilon/step5k/eps0/weight_lambda1e-3.json",
#     "config/hyperparam_v7/puddlerand/offline_learning/fqi-linear/fqi-adam/step5k/eps0/weight_lambda1e-3.json",
# ] , prev=0, line_per_file=1) # 20 min / 3 per hour

# write_grid_search(30, 30, [
#     # "config/hyperparam_v7/acrobot/offline_learning/fqi/fqi-adam/alpha_hidden_epsilon/step5k/eps0/online_lambda1e-3.json",
#     # "config/hyperparam_v7/acrobot/offline_learning/fqi-linear/fqi-adam/step5k/eps0/online_lambda1e-3.json",
#     # "config/hyperparam_v7/acrobot_shift/offline_learning/fqi/fqi-adam/alpha_hidden_epsilon/step5k/eps0/online_lambda1e-3.json",
#     # "config/hyperparam_v7/acrobot_shift/offline_learning/fqi-linear/fqi-adam/step5k/eps0/online_lambda1e-3.json",
#     "config/hyperparam_v7/acrobot_act_flip/offline_learning/fqi/fqi-adam/alpha_hidden_epsilon/step5k/eps0/online_lambda1e-3.json",
#     "config/hyperparam_v7/acrobot_act_flip/offline_learning/fqi-linear/fqi-adam/step5k/eps0/online_lambda1e-3.json",
#     # "config/hyperparam_v7/puddlerand/offline_learning/fqi/fqi-adam/alpha_hidden_epsilon/step5k/eps0/online_lambda1e-3.json",
#     # "config/hyperparam_v7/puddlerand/offline_learning/fqi-linear/fqi-adam/step5k/eps0/online_lambda1e-3.json",
# ] , prev=0, line_per_file=1) # 20 min / 3 per hour

# write_script(start_script=0, num_script=1, start_task=0, total_tasks=3240,
#              hours=3, min_node=1, parallel=36) #start_script, num_script, start_task, total_tasks, hours, min_node 45min
# write_script(start_script=0, num_script=1, start_task=4000, total_tasks=3240,
#              hours=3, min_node=1, parallel=40) #start_script, num_script, start_task, total_tasks, hours, min_node 45min

# for fi in range(1, 30):
#     write_cem(fi, parallel=20, hours=8)

write_bayes_opt(["config/hyperparam_v5/acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step500_env/data_suboptimal/drop0/bayesopt_template.json",
                 "config/hyperparam_v5/puddlerand/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step500_env/data_suboptimal/drop0/bayesopt_template.json"],
                start_script=0, num_cpu=5, num_inner_run=5, num_data=30)