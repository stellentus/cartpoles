import json
import argparse
import os
import numpy as np
import pandas as pd
import copy
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events


# def black_box_function(x, y):
#     """Function with unknown internals we wish to maximize.
#
#     This is just serving as an example, for all intents and
#     purposes think of the internals of this function, i.e.: the process
#     which generates its output values, as unknown.
#     """
#     return -x ** 2 - (y - 1) ** 2 + 1
class RunGoFunc:
    def __init__(self, cfg_template, num_runs, data_idx, max_cpu):
        self.counter = 0
        self.cfg_template = cfg_template
        self.num_runs = num_runs
        self.data_idx = data_idx
        self.path = cfg_template['config-path']
        self.outer = cfg_template["agent-settings"]["total-logs"]
        self.max_cpu = max_cpu

    def write_json(self, template, seed, sweep_dict, path):
        config_dict = copy.deepcopy(template)
        # config_dict["environment-settings"]["datalog"] = config_dict["environment-settings"]["datalog"].format(self.data_idx)
        # config_dict["environment-settings"]["rep-load-path"] = config_dict["environment-settings"]["rep-load-path"].format(self.data_idx)
        for key, value in sweep_dict.items():
            config_dict["agent-settings"]["sweep"][key] = [value]
        config_dict["experiment-settings"]["data-path"] = config_dict["experiment-settings"]["data-path"].format(self.data_idx, seed)

        with open(path, 'w') as conf:
            json.dump(config_dict, conf, indent=4)
        return config_dict["experiment-settings"]["data-path"]

    def loading_perf(self, data_path, source="return", outer=30):
        data_path = os.path.join(data_path, "param_0") # only 1 hyper setting in total
        all_perf = []
        for file in os.listdir(data_path):
            if ".csv" in file:
                fp = os.path.join(data_path, file)
                t_rwd = pd.read_csv(fp)["total reward"][0]
                num_ep = pd.read_csv(fp)[" total episodes"][0]
                if source=="return":
                    res = t_rwd / num_ep
                else:
                    raise NotImplementedError
                all_perf.append(res)
        return np.array(all_perf).mean()

    def single_setting(self, **hypers):
        new_conf_file_c = os.path.join(self.path, "dataset_{}_param_{}.json".format(self.data_idx, self.counter))
        data_path = self.write_json(self.cfg_template, self.counter, hypers, new_conf_file_c)
        self.counter += 1
        # for run in range(self.num_runs):
        #     os.system('./main --config {} --run {}'.format(new_conf_file_c, run))
        os.system('parallel --jobs '+str(self.max_cpu)+' ./main --config '+new_conf_file_c+' --run {} ::: $(seq '+str(self.data_idx)+' '+str(self.outer)+' '+str(self.num_runs*self.outer-1)+')')
        perf = self.loading_perf(data_path, source="return", outer=self.outer)
        return perf


if __name__ == '__main__':

    # Bounded region of parameter space
    # pbounds = {'x': (2, 4), 'y': (-3, 3)}
    #
    # optimizer = BayesianOptimization(
    #     f=black_box_function,
    #     pbounds=pbounds,
    #     random_state=1,
    # )
    # optimizer.maximize(
    #     init_points=2,
    #     n_iter=3,
    # )


    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--num-runs', default=5, type=int, help='number of runs')
    parser.add_argument('--log-idx', default=0, type=int, help='datalog index')
    parser.add_argument('--config-file', default='experiment/config/test_v0/dosing_prototype.json')
    parser.add_argument('--max_cpu', default=1, type=int, help='the number of cpus you can use')
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        cfg = json.load(f)

    pbounds = cfg['agent-settings']['sweep']
    if not os.path.isdir(cfg['config-path']):
        os.makedirs(cfg['config-path'])
    rgf = RunGoFunc(cfg, args.num_runs, args.log_idx, args.max_cpu)

    optimizer = BayesianOptimization(
        f=rgf.single_setting,
        pbounds=pbounds,
        random_state=args.log_idx,
    )

    fname = "bayes_env{}_dataset{}".format(cfg["agent-settings"]["env-name"], args.log_idx)
    logger = JSONLogger(path=fname+".json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=5,
        n_iter=100,
    )
    print(optimizer.max)
    with open(fname+"_max.txt", "w") as f:
        content = optimizer.max
        for key, value in content.items():
            f.write("{}:{}\n".format(key, value))