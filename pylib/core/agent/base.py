import numpy as np
import torch
import copy
from core.utils import torch_utils


class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.episode_reward = 0
        self.episode_rewards = []
        self.total_steps = 0
        self.reset = True
        self.ep_steps = 0
        self.num_episodes = 0
        self.timeout = cfg.timeout
        self.batch_indices = torch.arange(self.cfg.batch_size).long().to(cfg.device)
        self.eval_env = copy.deepcopy(cfg.env_fn())
        self.ep_returns_queue_train = np.zeros(cfg.stats_queue_size)
        self.ep_returns_queue_test = np.zeros(cfg.stats_queue_size)
        self.train_stats_counter = 0
        self.test_stats_counter = 0
        self.agent_rng = np.random.RandomState(self.cfg.seed)
        self.test_rng = np.random.RandomState(self.cfg.seed)
        self.replay = None
        self.true_q_predictor = self.cfg.tester_fn.get('true_value_estimator', lambda x:None)

    def update_stats(self, reward, done):
        self.episode_reward += reward
        self.total_steps += 1
        self.ep_steps += 1
        # print(self.ep_steps, self.total_steps, done)
        if done or self.ep_steps == self.timeout:
            self.episode_rewards.append(self.episode_reward)
            self.num_episodes += 1
            if self.cfg.evaluation_criteria == "return":
                self.add_train_log(self.episode_reward)
            elif self.cfg.evaluation_criteria == "steps":
                self.add_train_log(self.ep_steps)
            else:
                raise NotImplementedError
            self.episode_reward = 0
            self.ep_steps = 0
            self.reset = True

    def add_train_log(self, ep_return):
        self.ep_returns_queue_train[self.train_stats_counter] = ep_return
        self.train_stats_counter += 1
        self.train_stats_counter = self.train_stats_counter % self.cfg.stats_queue_size

    def add_test_log(self, ep_return):
        self.ep_returns_queue_test[self.test_stats_counter] = ep_return
        self.test_stats_counter += 1
        self.test_stats_counter = self.test_stats_counter % self.cfg.stats_queue_size

    def populate_returns(self, log_traj=False, total_ep=None, initialize=False):
        total_ep = self.cfg.stats_queue_size if total_ep is None else total_ep
        total_steps = 0
        total_states = []
        total_actions = []
        total_returns = []
        for ep in range(total_ep):
            ep_return, steps, traj = self.eval_episode(log_traj=log_traj)
            total_steps += steps
            total_states += traj[0]
            total_actions += traj[1]
            total_returns += traj[2]
            if self.cfg.evaluation_criteria == "return":
                self.add_test_log(ep_return)
                if initialize:
                    self.add_train_log(ep_return)
            elif self.cfg.evaluation_criteria == "steps":
                self.add_test_log(steps)
                if initialize:
                    self.add_train_log(steps)
            else:
                raise NotImplementedError
        # log_str = 'POPULATE LOG: total steps %d, total episodes %d'
        # self.cfg.logger.info(log_str % (total_steps, self.cfg.stats_queue_size))
        return [total_states, total_actions, total_returns]
    
    # hacking environment and policy
    def populate_returns_random_start(self, start_pos=None, start_policy=None, total_ep=None):
        total_ep = self.cfg.stats_queue_size if total_ep is None else total_ep
        total_states = []
        total_actions = []
        total_returns = []
        for ep in range(total_ep):
            s, a, ret = self.eval_episode_random_start(start_pos=start_pos, random_insert_policy=start_policy)
            total_states.append(s)
            total_actions.append(a)
            total_returns.append(ret)
        return [total_states, total_actions, total_returns]

    def random_fill_buffer(self, total_steps):
        state = self.eval_env.reset()
        for _ in range(total_steps):
            action = self.agent_rng.randint(0, self.cfg.action_dim)
            last_state = state
            state, reward, done, _ = self.eval_env.step([action])
            self.replay.feed([last_state, action, reward, state, int(done)])
            if done:
                state = self.eval_env.reset()
                # print("Done")

    def eval_episode(self, log_traj=False):
        ep_traj = []
        state = self.eval_env.reset()
        total_rewards = 0
        ep_steps = 0
        done = False
        while True:
            action = self.eval_step(state)
            last_state = state
            state, reward, done, _ = self.eval_env.step([action])
            if log_traj:
                ep_traj.append([last_state, action, reward])
            total_rewards += reward
            ep_steps += 1
            if done or ep_steps == self.cfg.timeout:
                break

        states = []
        actions = []
        rets = []
        if log_traj:
            # s, a, r = ep_traj[len(ep_traj)-1]
            # ret = r if done else self.true_q_predictor(self.cfg.state_normalizer(s))[a]
            # states = [s]
            # actions = [a]
            # rets = [ret]
            # for i in range(len(ep_traj)-2, -1, -1):
            ret = 0
            for i in range(len(ep_traj)-1, -1, -1):
                s, a, r = ep_traj[i]
                ret = r + self.cfg.discount * ret
                rets.insert(0, ret)
                actions.insert(0, a)
                states.insert(0, s)
        return total_rewards, ep_steps, [states, actions, rets]

    # hacking environment and policy
    def eval_episode_random_start(self, start_pos, random_insert_policy):
        if start_pos is None:
            ep_traj = []
            state = self.eval_env.reset()
            total_rewards = 0
            ep_steps = 0
            while True:
                action = self.eval_step(state)
                last_state = state
                state, reward, done, _ = self.eval_env.step([action])
                ep_traj.append([last_state, action, reward])
                total_rewards += reward
                ep_steps += 1
                if done or ep_steps == self.cfg.timeout:
                    break
            
            random_idx = self.test_rng.randint(len(ep_traj))
            random_start = ep_traj[random_idx][0]
        else:
            random_start = start_pos

        action = random_insert_policy(random_start)
        state, reward, done, _ = self.eval_env.hack_step(random_start, action)
        if done:
            return random_start, action, reward
        ep_traj = []
        total_rewards = 0
        ep_steps = 0
        while True:
            action = self.eval_step(state)
            last_state = state
            state, reward, done, _ = self.eval_env.hack_step(last_state, action)
            ep_traj.append([last_state, action, reward])
            total_rewards += reward
            ep_steps += 1
            if done or ep_steps == self.cfg.timeout:
                break

        ret = 0
        for i in range(len(ep_traj)-1, -1, -1):
            s, a, r = ep_traj[i]
            ret = r + self.cfg.discount * ret
        return s, a, ret

    def eval_episodes(self):
        return

    def log_return(self, log_ary, name, elapsed_time):
        rewards = log_ary
        total_episodes = len(self.episode_rewards)
        mean, median, min_, max_ = np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards)

        log_str = '%s LOG: steps %d, episodes %3d, ' \
                  'returns %.2f/%.2f/%.2f/%.2f/%d (mean/median/min/max/num), %.2f steps/s'

        self.cfg.logger.info(log_str % (name, self.total_steps, total_episodes, mean, median,
                                        min_, max_, len(rewards),
                                        elapsed_time))
        return mean, median, min_, max_

    def log_file(self, elapsed_time=-1):
        mean, median, min_, max_ = self.log_return(self.ep_returns_queue_train, "TRAIN", elapsed_time)
        ##Check bug
        # if self.total_steps > 0:
        #     print(self.env.reset())
        #     print(self.env.state)
        #     print(self.env.step([0]))
        #     print("Right before", self.env.state)
        #     self.populate_returns()
        #     print("Right after", self.env.state)
        #     print("Right after", self.eval_env.state)
        #     print(self.env.step([0]))
        #     exit()
        self.populate_returns()
        mean, median, min_, max_ = self.log_return(self.ep_returns_queue_test, "TEST", elapsed_time)
        return mean, median, min_, max_

    def policy(self, state, eps):
        raise NotImplementedError

    def eval_step(self, state):
        # action = self.policy(state, 0)
        # return action
        raise NotImplementedError

    def save(self, filename):
        raise NotImplementedError

    def load(self, filename):
        raise NotImplementedError

    def one_hot_action(self, actions):
        one_hot = np.zeros((len(actions), self.cfg.action_dim))
        np.put_along_axis(one_hot, actions.reshape((-1, 1)), 1, axis=1)
        return one_hot

    # def log_overestimation(self):
    #     return

    def training_set_construction(self, data_dict, value_predictor=None):
        if value_predictor is None:
            value_predictor = lambda x: self.val_net(self.rep_net(x))
            
        states = []
        actions = []
        rewards = []
        next_states = []
        terminations = []
        next_actions = []
        qmaxs = []
    
        for name in data_dict:
            states.append(data_dict[name]['states'])
            actions.append(data_dict[name]['actions'])
            rewards.append(data_dict[name]['rewards'])
            next_states.append(data_dict[name]['next_states'])
            terminations.append(data_dict[name]['terminations'])
            next_actions.append(np.concatenate([data_dict[name]['actions'][1:], [-1]]))  # Should not be used when using the current estimation in target construction
            if 'qmax' in data_dict[name].keys():
                qmaxs.append(data_dict[name]['qmax'])

        states = np.concatenate(states)
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards)
        next_states = np.concatenate(next_states)
        terminations = np.concatenate(terminations)
        next_actions = np.concatenate(next_actions)
        if len(qmaxs) > 0:
            qmaxs = np.concatenate(qmaxs)
        for i in range(len(states)):
            states[i] = self.cfg.state_normalizer(states[i])
            next_states[i] = self.cfg.state_normalizer(next_states[i])
    
        pred_returns = np.zeros(len(states))
        true_returns = np.zeros(len(states))
        for i in range(len(states) - 1, -1, -1):
            if i == len(states) - 1 or (not np.array_equal(next_states[i], states[i + 1])):
                pred_returns[i] = value_predictor(self.cfg.state_normalizer(states[i]))[actions[i]]
                true_pred = self.true_q_predictor(self.cfg.state_normalizer(states[i]))
                true_returns[i] = 0 if true_pred is None else true_pred[actions[i]]
            else:
                end = 1.0 if terminations[i] else 0.0
                pred_returns[i] = rewards[i] + (1 - end) * self.cfg.discount * pred_returns[i + 1]
                true_returns[i] = rewards[i] + (1 - end) * self.cfg.discount * true_returns[i + 1]

        thrshd = int(len(states) * 1)
    
        training_s = states[: thrshd]
        training_a = actions[: thrshd]
        training_r = rewards[: thrshd]
        training_ns = next_states[: thrshd]
        training_t = terminations[: thrshd]
        training_na = next_actions[: thrshd]
        training_pred_ret = pred_returns[: thrshd]
        training_true_ret = true_returns[: thrshd]
        training_qmax = qmaxs[: thrshd]

        testing_s = states[thrshd:]
        testing_a = actions[thrshd:]
        testing_r = rewards[thrshd:]
        testing_ns = next_states[thrshd:]
        testing_t = terminations[thrshd:]
        testing_na = next_actions[thrshd:]
        testing_pred_ret = pred_returns[thrshd:]
        testing_true_ret = true_returns[thrshd:]
        testing_qmax = qmaxs[thrshd:]

        return [np.array(training_s), training_a, np.array(training_r), np.array(training_ns), np.array(training_t), training_na, training_pred_ret, training_true_ret, training_qmax], \
               [np.array(testing_s), testing_a, np.array(testing_r), np.array(testing_ns), np.array(testing_t), testing_na, testing_pred_ret, testing_true_ret, testing_qmax]

    def property_evaluation_dataset(self, data_dict):
        states = []
        actions = []
        returns = []
        for name in data_dict:
            states.append(data_dict[name]['states'])
            actions.append(data_dict[name]['actions'])
            returns.append(data_dict[name]['returns'])
        if len(states) > 0:
            states = np.concatenate(states)
            actions = np.concatenate(actions)
            returns = np.concatenate(returns)
        return [states, actions, returns]

    # def dataset_to_episodes(self, data_dict):
    #     episodes = []
    #     for name in data_dict:
    #         states = data_dict[name]['states']
    #         actions = data_dict[name]['actions']
    #         rewards = data_dict[name]['rewards']
    #         next_states = data_dict[name]['next_states']
    #         terminations = data_dict[name]['terminations']
    #
    #         ep = []
    #         for i in range(len(terminations)):
    #             s, a, r, t = states[i], actions[i], rewards[i], terminations[i]
    #             ep.append([s, a, r, t])
    #             if t or i == len(terminations) - 1 or not np.array_equal(next_states[i], states[i + 1]):
    #                 episodes.append(ep)
    #                 ep = []
    #     return episodes
    #
    # def target_update(self, episodes):
    #     input_s_all, input_a_all, target_all, true_q_all = self.training_set_construction(episodes)
    #     idxs = np.arange(len(input_s_all))
    #     # self.agent_rng.shuffle(idxs) # make sure the test set keeps the same each time
    #     thrshd = int(len(idxs) * 0.8)
    #     self.training_input_s, self.training_input_a, self.training_target = input_s_all[idxs[: thrshd]], input_a_all[idxs[: thrshd]], target_all[idxs[: thrshd]]
    #     self.test_input_s, self.test_input_a, self.test_target = input_s_all[idxs[thrshd:]], input_a_all[idxs[thrshd:]], target_all[idxs[thrshd:]]
    #     self.training_size = len(self.training_input_s)
    #     self.training_indexs = np.arange(self.training_size)
    #
    #
    # def training_set_construction(self, episodes, value_predictor=None):
    #     if value_predictor is None:
    #         value_predictor = lambda x: self.val_net(self.rep_net(x))
    #     states = []
    #     states_normalized = []
    #     actions = []
    #     rewards = []
    #     next_states = []
    #     terminates = []
    #     next_actions = []
    #     dataset_return = []
    #     true_return = []
    #     for ep in episodes:
    #         ret = 0
    #         ret_true = 0
    #         for i in range(len(ep) - 1, -1, -1):
    #             s, a, r, t = ep[i]
    #             if i == len(ep) - 1 and ep[i][3] == 0:
    #                 ret_true = self.true_q_predictor(self.cfg.state_normalizer(s))[a]
    #                 in_ = torch_utils.tensor(self.cfg.state_normalizer(s), self.cfg.device)
    #                 ret = torch_utils.to_np(value_predictor(in_)[a])
    #             elif i == len(ep) - 1 and ep[i][3] == 1:
    #                 ret_true = 0
    #                 ret = 0
    #             else:
    #                 ret_true = ret_true * self.cfg.discount + r
    #                 ret = ret * self.cfg.discount + r
    #             states.append(s)
    #             states_normalized.append(self.cfg.state_normalizer(s))
    #             actions.append(a)  # used as index
    #             rewards.append(r)
    #             terminates.append(t)
    #             dataset_return.append([ret])
    #             true_return.append([ret_true])
    #     return np.array(states), np.array(actions), np.array(dataset_return), np.array(true_return)

    def log_overestimation(self):
        # def get_true_q_value(states, actions):
        #     all_returns = np.zeros((len(states)))
        #     for i, (state, action) in enumerate(zip(states, actions)):
        #         _, _, returns = self.populate_returns_random_start(state, lambda x: action, total_ep=5)
        #         all_returns[i] = np.array(returns).mean()
        #     return all_returns
    
        # true_qs = get_true_q_value(test_s, test_a)
        test_s, test_a, true_ret = self.eval_set
        with torch.no_grad():
            q_values = self.val_net(self.rep_net(torch_utils.tensor(self.cfg.state_normalizer(test_s), self.cfg.device)))
            q_values = torch_utils.to_np(q_values)
        onpolicy_q = q_values[np.arange(len(q_values)), test_a]
        all_diff = onpolicy_q - true_ret
        log_str = 'TRAIN LOG: epoch %d, ' \
                  'Overestimation: %.8f/%.8f/%.8f (mean/min/max)'
        self.cfg.logger.info(log_str % (self.total_steps, all_diff.mean(), all_diff.min(), all_diff.max()))
        
    def log_overestimation_current_pi(self):
        states, actions, true_qs = self.populate_returns(log_traj=True)
        states = np.array(states)
        true_qs = np.array(true_qs)
        with torch.no_grad():
            phis = self.rep_net(self.cfg.state_normalizer(states))
            q_values = self.val_net(phis)
            q_values = torch_utils.to_np(q_values)
        onpolicy_q = q_values[np.arange(len(q_values)), actions]
        all_diff = onpolicy_q - true_qs
        log_str = 'TRAIN LOG: epoch %d, ' \
                  'OverestimationCurrentPi: %.8f/%.8f/%.8f (mean/min/max)'
        self.cfg.logger.info(log_str % (self.total_steps, all_diff.mean(), all_diff.min(), all_diff.max()))
