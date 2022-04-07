import copy
import json
import numpy as np

def sample(low, high, rng):
    return rng.uniform(low, high)

def write_json(template, seed, range_dict, path):
    rng = np.random.RandomState(seed)
    config_dict = copy.deepcopy(template)
    for key, r in range_dict.items():
        value = sample(r[0], r[1], rng)
        config_dict["agent-settings"]["sweep"][key] = [value]
    config_dict["experiment-settings"]["data-path"] = config_dict["experiment-settings"]["data-path"].format(seed)

    new_conf_file_c = path.format(seed)
    with open(new_conf_file_c, 'w') as conf:
        json.dump(config_dict, conf, indent=4)

def acrobot_5k_opt():
    template = {
        "agent-name": "esarsa",
        "environment-name": "knnModel",
        "agent-settings": {
            "gamma": 1.0,
            "state-len": 6,
            "env-name": "acrobot",
            "sweep": {
                "tilings": [16],
                "tiles": [8],
                "is-stepsize-adaptive": [True],
                "alpha": [0.0],
                "lambda": [0.8],
                "epsilon": [0.0],
                "adaptive-alpha": [],
                "beta1": [0.0],
                "softmax-temp": [],
                "weight-init": []
            },
            "lock-weight": False,
            "enable-debug": False,
            "seed": 1,

            "total-logs": 30
        },
        "environment-settings": {
            "seed": 1,
            "total-logs": 30,
            "neighbor-num": 3,
            "datalog": "data/hyperparam_v5/acrobot/offline_data/random_restarts/esarsa/step5k/optimalfixed_eps0/param_0/",
            "true-start-log": "data/hyperparam_v5/acrobot/offline_data/true_restarts/esarsa/step5k/optimalfixed_eps0/param_0/",
            "ensemble-seed": 0,
            "drop-percent": 0,
            "pick-start-state": "random-init",

            "rep-train-num-step": 30000,
            "rep-train-beta": 5,
            "rep-train-delta": 0.5,
            "rep-train-lambda": 0.8,
            "rep-train-traj-len": 20,
            "rep-train-batch": 128,
            "rep-train-learning-rate": 0.00003,
            "rep-hidden": [128, 128],
            "rep-dim": 8,
            "rep-name": "Laplace",

            "rep-save": False,
            "rep-load": True,
            "rep-load-path": "data/hyperparam_v5/acrobot/offline_learning/knn/env_training/step5k/optimalfixed_eps0/rep_laplace"
        },
        "experiment-settings": {
            "randomize_start_state_beforeLock": False,
            "randomize_start_state_afterLock": False,
            "steps": 0,
            "episodes": 1000000,
            "max-run-length-episodic": 15000,
            "steps-in-episode": 500,

            "data-path": "data/hyperparam_v5/acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step5k_env/data_optimal/drop0/gridsearch_uniform_sample/param_{}",
            "should-log-traces": False,
            "should-log-episode-lengths": False,
            "should-log-rewards": False,
            "should-log-totals": True,
            "should-log-returns": False,
            "debug-interval": 0
        }
    }
    range_dict = {
        "adaptive-alpha": [0.003, 0.3],
        "softmax-temp": [1, 100],
        "weight-init": [0, 16]
    }
    path = "config/hyperparam_v5/acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step5k_env/data_optimal/drop0/gridsearch_uniform_sample/param_{}.json"
    for seed in range(54): # 54 parameters, 30 datasets, 10 runs on calibration model each
        write_json(template, seed, range_dict, path)

def puddle_world_5k_opt():
    template = {
        "agent-name": "esarsa",
        "environment-name": "knnModel",
        "agent-settings": {
            "gamma": 1.0,
            "state-len": 2,
            "env-name": "puddleworld",
            "sweep": {
                "tilings": [16],
                "tiles": [8],
                "is-stepsize-adaptive": [True],
                "alpha": [0.0],
                "lambda": [0.1],
                "epsilon": [0.0],
                "adaptive-alpha": [],
                "beta1": [0.0],
                "softmax-temp": [],
                "weight-init": []
            },
            "lock-weight": False,
            "enable-debug": False,
            "seed": 1,
            "total-logs": 30
        },
        "environment-settings": {
            "seed": 1,
            "total-logs": 30,
            "neighbor-num": 3,
            "datalog": "data/hyperparam_v5/puddlerand/offline_data/random_restarts/esarsa/step5k/optimalfixed_eps0/param_0/",
            "ensemble-seed": 0,
            "drop-percent": 0,
            "pick-start-state": "random-init",

            "rep-train-num-step": 30000,
            "rep-train-beta": 5,
            "rep-train-delta": 0.05,
            "rep-train-lambda": 0.8,
            "rep-train-traj-len": 10,
            "rep-train-batch": 128,
            "rep-train-learning-rate": 0.0003,
            "rep-hidden": [128, 128],
            "rep-dim": 4,
            "rep-test-forward": 1,
            "rep-name": "Laplace",

            "rep-save": False,
            "rep-load": True,
            "rep-load-path": "data/hyperparam_v5/puddlerand/offline_learning/knn/env_training/step5k/optimalfixed_eps0/rep_laplace"
        },
        "experiment-settings": {
            "randomize_start_state_beforeLock": False,
            "randomize_start_state_afterLock": False,
            "steps": 0,
            "episodes": 1000000,
            "max-run-length-episodic": 30000,
            "steps-in-episode": 1000,

            "data-path": "data/hyperparam_v5/puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step5k_env/data_optimal/drop0/gridsearch_uniform_sample/param_{}",
            "should-log-traces": False,
            "should-log-episode-lengths": False,
            "should-log-rewards": False,
            "should-log-totals": True,
            "should-log-returns": False,
            "debug-interval": 0
        }
    }
    range_dict = {
        "adaptive-alpha": [0.01, 0.1],
        "softmax-temp": [1.0, 100.0],
        "weight-init": [0, 8]
    }
    path = "config/hyperparam_v5/puddlerand/offline_learning/knn/learning/k3_laplace/timeout1000/esarsa/step5k_env/data_optimal/drop0/gridsearch_uniform_sample/param_{}.json"
    for seed in range(54): # 54 parameters, 30 datasets, 10 runs on calibration model each
        write_json(template, seed, range_dict, path)


def acrobot_500_subopt():
    template = {
        "agent-name": "esarsa",
        "environment-name": "knnModel",
        "agent-settings": {
            "gamma": 1.0,
            "state-len": 6,
            "env-name": "acrobot",
            "sweep": {
                "tilings": [16],
                "tiles": [8],
                "is-stepsize-adaptive": [True],
                "alpha": [0.0],
                "lambda": [0.8],
                "epsilon": [0.0],
                "adaptive-alpha": [],
                "beta1": [0.0],
                "softmax-temp": [],
                "weight-init": [0.0]
            },
            "lock-weight": False,
            "enable-debug": False,
            "seed": 1,

            "total-logs": 30
        },
        "environment-settings": {
            "seed": 1,
            "total-logs": 30,
            "neighbor-num": 3,
            "datalog": "data/hyperparam_v5/acrobot/offline_data/random_restarts/esarsa/step500/suboptimalfixed_eps0/param_0/",
            "true-start-log": "data/hyperparam_v5/acrobot/offline_data/true_restarts/esarsa/step500/suboptimalfixed_eps0/param_0/",
            "ensemble-seed": 0,
            "drop-percent": 0,
            "pick-start-state": "random-init",

            "rep-train-num-step": 30000,
            "rep-train-beta": 5,
            "rep-train-delta": 0.5,
            "rep-train-lambda": 0.8,
            "rep-train-traj-len": 20,
            "rep-train-batch": 128,
            "rep-train-learning-rate": 0.00003,
            "rep-hidden": [128, 128],
            "rep-dim": 8,
            "rep-test-forward": 1,
            "rep-name": "Laplace",

            "rep-save": False,
            "rep-load": True,
            "rep-load-path": "data/hyperparam_v5/acrobot/offline_learning/knn/env_training/step500/suboptimalfixed_eps0/rep_laplace"
        },
        "experiment-settings": {
            "randomize_start_state_beforeLock": False,
            "randomize_start_state_afterLock": False,
            "steps": 0,
            "episodes": 1000000,
            "max-run-length-episodic": 15000,
            "steps-in-episode": 500,

            "data-path": "data/hyperparam_v5/acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step500_env/data_suboptimal/drop0/cem_rand_sample/param_{}",
            "should-log-traces": False,
            "should-log-episode-lengths": False,
            "should-log-rewards": False,
            "should-log-totals": True,
            "should-log-returns": False,
            "debug-interval": 0
        }
    }
    range_dict = {
        "adaptive-alpha": [0, 0.1],
        "softmax-temp": [0.0001, 5]
    }
    path = "config/hyperparam_v5/acrobot/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step500_env/data_suboptimal/drop0/cem_uniform_sample/param_{}.json"
    for seed in range(100):
        write_json(template, seed, range_dict, path)

def puddle_world_500_subopt():
    template = {
        "agent-name": "esarsa",
        "environment-name": "knnModel",
        "agent-settings": {
            "gamma": 1.0,
            "state-len": 2,
            "env-name": "puddleworld",
            "sweep": {
                "tilings": [16],
                "tiles": [8],
                "is-stepsize-adaptive": [True],
                "alpha": [0.0],
                "lambda": [0.1],
                "epsilon": [0.0],
                "adaptive-alpha": [],
                "beta1": [0.0],
                "softmax-temp": [],
                "weight-init": [0.0]
            },
            "lock-weight": False,
            "enable-debug": False,
            "seed": 1,
            "total-logs": 30
        },
        "environment-settings": {
            "seed": 1,
            "total-logs": 30,
            "neighbor-num": 3,
            "datalog": "data/hyperparam_v5/puddlerand/offline_data/random_restarts/esarsa/step500/suboptimalfixed_eps0/param_0/",
            "ensemble-seed": 0,
            "drop-percent": 0,
            "pick-start-state": "random-init",

            "rep-train-num-step": 30000,
            "rep-train-beta": 5,
            "rep-train-delta": 0.05,
            "rep-train-lambda": 0.8,
            "rep-train-traj-len": 10,
            "rep-train-batch": 128,
            "rep-train-learning-rate": 0.0003,
            "rep-hidden": [128, 128],
            "rep-dim": 4,
            "rep-test-forward": 1,
            "rep-name": "Laplace",

            "rep-save": False,
            "rep-load": True,
            "rep-load-path": "data/hyperparam_v5/puddlerand/offline_learning/knn/env_training/step500/suboptimalfixed_eps0/rep_laplace"
        },
        "experiment-settings": {
            "randomize_start_state_beforeLock": False,
            "randomize_start_state_afterLock": False,
            "steps": 0,
            "episodes": 1000000,
            "max-run-length-episodic": 15000,
            "steps-in-episode": 500,

            "data-path": "data/hyperparam_v5/puddlerand/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step500_env/data_suboptimal/drop0/cem_rand_sample/param_{}",
            "should-log-traces": False,
            "should-log-episode-lengths": False,
            "should-log-rewards": False,
            "should-log-totals": True,
            "should-log-returns": False,
            "debug-interval": 0
        }
    }
    range_dict = {
        "adaptive-alpha": [0, 0.1],
        "softmax-temp": [0.001, 10]
    }
    path = "config/hyperparam_v5/puddlerand/offline_learning/knn/learning/k3_laplace/timeout500/esarsa/step500_env/data_suboptimal/drop0/cem_uniform_sample/param_{}.json"
    for seed in range(100):
        write_json(template, seed, range_dict, path)

if __name__ == '__main__':
    acrobot_5k_opt()
    puddle_world_5k_opt()
    # acrobot_500_subopt()
    # puddle_world_500_subopt()
