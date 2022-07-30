import os.path
import pickle
import time
import copy
import numpy as np
import pandas as pd
from core.utils.torch_utils import ensure_dir

EARLYCUTOFF = "EarlyCutOff"
# def load_testset(paths, run=0):
#     if paths is not None:
#         testsets = {}
#         for name in paths:
#             if name == "buffer":
#                 testsets[name] = {
#                     'states': None,
#                     'actions': None,
#                     'rewards': None,
#                     'next_states': None,
#                     'terminations': None,
#                 }
#             elif name == "diff_pi":
#                 pth = paths[name]
#                 with open(pth.format(run), 'rb') as f:
#                     pairs = pickle.load(f)
#                 testsets[name] = {
#                     'states': pairs["state0"],
#                     'actions': None,
#                     'rewards': None,
#                     'next_states': pairs["state1"],
#                     'terminations': None,
#                 }
#             else:
#                 pth = paths[name]
#                 with open(pth.format(run), 'rb') as f:
#                     testsets[name] = pickle.load(f)
#
#                 print(np.array([testsets[name]['next_states'][i] - testsets[name]['states'][i] for i in range(len(testsets[name]['next_states']))]).mean(axis=0))
#                 exit()
#                 # terminal = testsets[name]['terminations']
#                 # s_ary = testsets[name]['states']
#                 # idx = np.where(terminal == 1)[0]
#                 # states = s_ary[idx]
#                 # import matplotlib.pyplot as plt
#                 # fig, axs = plt.subplots(2,1)
#                 # angle_states = np.zeros((len(states), 4))
#                 # for i,s in enumerate(states):
#                 #     angle_states[i] = np.array([arcradians(s[0], s[1]),
#                 #                                 arcradians(s[2], s[3]),
#                 #                                 s[4],
#                 #                                 s[5]])
#                 # sid = np.arange(0, len(states))
#                 # axs[0].scatter(angle_states[sid, 0], angle_states[sid, 1], s=1)
#                 # axs[1].scatter(angle_states[sid, 2], angle_states[sid, 3], s=1)
#                 # axs[0].set_xlim((-3.2, 3.2))
#                 # axs[0].set_ylim((-3.2, 3.2))
#                 # axs[1].set_xlim((-13, 13))
#                 # axs[1].set_ylim((-30, 30))
#                 # plt.show()
#         return testsets
#     else:
#         return {}

def arcradians(cos, sin):
    if cos > 0 and sin > 0:
        return np.arccos(cos)
    elif cos > 0 and sin < 0:
        return np.arcsin(sin)
    elif cos < 0 and sin > 0:
        return np.arccos(cos)
    elif cos < 0 and sin < 0:
        return -1 * np.arccos(cos)

def load_testset(paths, run=0):
    def str2array(s_str):
        s_ary = []
        for s in s_str:
            s_lst = s.strip("[").strip("]").split(" ")
            s_float = [float(f) for f in s_lst]
            s_ary.append(s_float)
        s_ary = np.array(s_ary)
        return s_ary

    testsets = {}
    if paths == {}:
        return testsets
    for name in paths:
        path = paths[name]
        testsets[name] = {}

        data = pd.read_csv(path.format(run))
        s_ary = str2array(data['previous state'].to_numpy())
        sp_ary = str2array(data['new state'].to_numpy())
        action = data['action'].to_numpy()
        reward = data['reward'].to_numpy()
        terminal = data['terminal'].to_numpy()

        testsets[name]['states'] = s_ary
        testsets[name]['actions'] = action
        testsets[name]['rewards'] = reward
        testsets[name]['next_states'] = sp_ary
        testsets[name]['terminations'] = terminal

        # print(np.array([sp_ary[i] - s_ary[i] for i in range(len(sp_ary))]).mean(axis=0))
        # exit()

        # idx = np.where(terminal == 1)[0]
        # states = s_ary[idx]
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(2,1)
        # angle_states = np.zeros((len(states), 4))
        # for i,s in enumerate(states):
        #     angle_states[i] = np.array([arcradians(s[0], s[1]),
        #                                 arcradians(s[2], s[3]),
        #                                 s[4],
        #                                 s[5]])
        # sid = np.arange(0, len(states))
        # axs[0].scatter(angle_states[sid, 0], angle_states[sid, 1], s=1)
        # axs[1].scatter(angle_states[sid, 2], angle_states[sid, 3], s=1)
        # axs[0].set_xlim((-3.2, 3.2))
        # axs[0].set_ylim((-3.2, 3.2))
        # axs[1].set_xlim((-13, 13))
        # axs[1].set_ylim((-30, 30))
        # plt.show()

        # import matplotlib
        # import matplotlib.pyplot as plt
        # def cmap4array(ary, min, max, getcm=False):
        #     cm = 'cool'
        #     getc = matplotlib.cm.get_cmap(cm)
        #     range_ = max - min
        #     if getcm:
        #         cm = matplotlib.cm.get_cmap(cm)
        #         norm = matplotlib.colors.Normalize(vmin=min, vmax=max)
        #         return getc(ary/range_), cm, norm
        #     return getc(ary/range_)
        # fig, axs = plt.subplots(1,1)
        # idx = np.where(reward < -1)[0]
        # states = s_ary#[idx]
        # reward = reward#[idx]
        # axs.plot([0.45, 0.1], [0.75, 0.75], color="black")
        # axs.plot([0.45, 0.45], [0.8, 0.4], color="black")
        # axs.scatter(states[:, 0], states[:, 1], s=1, c=cmap4array(reward, -45, -1))
        # axs.set_xlim(0, 1)
        # axs.set_ylim(0, 1)
        # plt.show()

    return testsets

def load_true_values(cfg):
    if cfg.true_value_paths is not None:
        valuesets = {}
        for name in cfg.true_value_paths:
            pth = cfg.true_value_paths[name]
            with open(pth, 'rb') as f:
                valuesets[name] = pickle.load(f)
        return valuesets
    else:
        return {}

def run_steps(agent):
    # valuesets = load_true_values(agent.cfg)
    t0 = time.time()
    transitions = []
    goto_states = []
    # agent.populate_returns(initialize=True)
    agent.random_fill_buffer(agent.cfg.warm_up_step)
    while True:
        if agent.cfg.log_interval and not agent.total_steps % agent.cfg.log_interval:
            if agent.cfg.tensorboard_logs: agent.log_tensorboard()
            mean, median, min, max = agent.log_file(elapsed_time=agent.cfg.log_interval / (time.time() - t0))
            t0 = time.time()

        if agent.cfg.eval_interval and not agent.total_steps % agent.cfg.eval_interval:
            # agent.eval_episodes(elapsed_time=agent.cfg.log_interval / (time.time() - t0))
            # agent.eval_episodes()
            if agent.cfg.visualize and agent.total_steps > 1:
                agent.visualize()
            if agent.cfg.evaluate_overestimation:
                agent.log_overestimation()
                agent.log_overestimation_current_pi()
            # t0 = time.time()
        if agent.cfg.max_steps and agent.total_steps >= agent.cfg.max_steps:
            break

        seq = agent.step()
        
        if agent.cfg.early_cut_off and seq == EARLYCUTOFF:
            break

        if agent.cfg.log_observations:
            transitions.append(copy.deepcopy(seq))
            # print(seq[3])
            goto_states.append(copy.deepcopy(seq[3]))

    if agent.cfg.save_params:
        agent.save()

    if agent.cfg.log_observations:
        import matplotlib.pyplot as plt
        goto_states = np.array(goto_states)
        plt.scatter(goto_states[:, 0], goto_states[:, 1])
        plt.show()
        # data_dir = agent.cfg.get_data_dir()
        # with open(os.path.join(data_dir, 'transition.pkl'), 'wb') as f:
        #     pickle.dump(transitions, f)

    if agent.cfg.save_csv:
        agent.episode_rewards.append(agent.episode_reward)
        agent.num_episodes += 1
        df = pd.DataFrame({'total reward': [np.array(agent.episode_rewards).sum()],
                           'total episodes': [len(agent.episode_rewards)]
                           })
        ensure_dir(agent.cfg.csv_path+'/param_{}'.format(agent.cfg.param_setting))
        df.to_csv(os.path.join(agent.cfg.csv_path, 'param_{}/totals-{}.csv'.format(agent.cfg.param_setting, agent.cfg.run)))


def value_iteration(env, gamma):
    max_iter = 10000
    done = False
    iter_count = 0
    eps = 0
    p_matrix, r_matrix, goal_idx, all_states = env.transition_reward_model()
    
    num_states = len(p_matrix)
    num_actions = len(env.actions)
    v_matrix = np.zeros(num_states)

    while not done and iter_count < max_iter:
        v_new = np.zeros(num_states)
        for i in range(num_states):
            for a in range(num_actions):
                cur_val = 0
                for j in np.nonzero(p_matrix[i][a])[0]:
                    cur_val += p_matrix[i][a][j] * v_matrix[j]
                if i == goal_idx:
                    cur_val *= 0.
                else:
                    cur_val *= gamma
                cur_val += r_matrix[i][a]
                v_new[i] = max(v_new[i], cur_val)
        max_diff = 0
        for i in range(num_states):
            max_diff = max(max_diff, abs(v_matrix[i] - v_new[i]))
        
        v_matrix = v_new
        
        iter_count += 1
        if (max_diff <= eps):
            print("state value converged at {}th iteration".format(iter_count))
            done = True
    
    
    q_matrix = np.zeros((num_states, num_actions))
    for i in range(num_states):
        for a in range(num_actions):
            temp = 0
            for j in np.nonzero(p_matrix[i][a])[0]:
                temp += p_matrix[i][a][j] * (r_matrix[i][a] + gamma * v_matrix[j])
            q_matrix[i, a] = temp
            
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, num_actions, figsize=(12, 3))
    # for a in range(num_actions):
    #     check = np.zeros((15, 15))
    #     for idx, s in enumerate(all_states):
    #         x, y = s
    #         check[x, y] = q_matrix[idx, a]
    #     im = axs[a].imshow(check, cmap="Blues", vmin=0.5, vmax=1, interpolation='none')
    # plt.colorbar(im)
    # plt.show()
    
    return q_matrix, all_states