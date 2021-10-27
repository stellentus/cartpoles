import subprocess
import time
import datetime
from multiprocessing import Pool


def run(cmd: str) -> None:
    print('>>', cmd)
    subprocess.call(cmd.split())


if __name__ == '__main__':

    print('build binary...')
    run('go build -o main cmd/experiment/main.go')

    tic = time.time()
    # commands = [
    #     f'./main --config config/hyperparam_v7/acrobot/offline_learning/fqi-linear/fqi-adam/step30k/optimal_data/weight_lambda1e-1.json --sweep {i}'
    #     for i in range(150)
    # ]
    # commands = [
    #     f'./main --config config/hyperparam_v7/puddlerand/offline_learning/fqi-linear/fqi-adam/step30k/optimal_data/weight_lambda1e-1.json --sweep {i}'
    #     for i in range(150)
    # ]

    # commands = [
    #     f'./main --config config/hyperparam_v7/puddlerand/offline_learning/fqi/fqi-adam/alpha_hidden_epsilon/step30k/optimal_data/weight_lambda1e-1.json --sweep {i}'
    #     for i in range(150)
    # ]
    #
    # commands += [
    #     f'./main --config config/hyperparam_v7/acrobot/offline_learning/fqi/fqi-adam/alpha_hidden_epsilon/step30k/optimal_data/weight_lambda1e-1.json --sweep {i}'
    #     for i in range(150)
    # ]

    commands = [
        f'./main --config config/hyperparam_v7/puddlerand/offline_learning/fqi/fqi-adam/alpha_hidden_epsilon/step30k/optimal_data/online_lambda1e-1.json --sweep {i}'
        for i in range(30)
    ]

    commands += [
        f'./main --config config/hyperparam_v7/acrobot/offline_learning/fqi/fqi-adam/alpha_hidden_epsilon/step30k/optimal_data/online_lambda1e-1.json --sweep {i}'
        for i in range(30)
    ]

    num_process = 3
    if num_process == 1:
        for cmd in commands:
            run(cmd)
    else:
        with Pool(num_process) as pool:
            pool.map(run, commands)

    print(f'Experiment takes {datetime.timedelta(seconds=time.time()-tic)}')
