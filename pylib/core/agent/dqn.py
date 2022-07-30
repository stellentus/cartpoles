import os
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import torch

from core.agent import base
from core.utils import torch_utils


class DQNAgent(base.Agent):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.rep_net = cfg.rep_fn()
        self.val_net = cfg.val_fn()

        # Creating target networks for value, representation, and auxiliary val_net
        rep_net_target = cfg.rep_fn()
        rep_net_target.load_state_dict(self.rep_net.state_dict())
        val_net_target = cfg.val_fn()
        val_net_target.load_state_dict(self.val_net.state_dict())

        # print(list(self.val_net.parameters())[-2])
        # exit()

        # Creating Target Networks
        TargetNets = namedtuple('TargetNets', ['rep_net', 'val_net'])
        self.targets = TargetNets(rep_net=rep_net_target, val_net=val_net_target)

        params = list(self.rep_net.parameters()) + list(self.val_net.parameters())
        self.optimizer = cfg.optimizer_fn(params)

        # self.independent_constr_opt = False
        # if cfg.constraint is not None and cfg.constraint['type'] == "L1":
        #     constr_param = list(self.rep_net.state_dict()['net.fc_head.weight']) + list(self.rep_net.state_dict()['net.fc_head.bias'])
        #     self.constr_optimizer = cfg.optimizer_fn(constr_param)
        #     self.independent_constr_opt = True

        if 'load_params' in self.cfg.rep_fn_config and self.cfg.rep_fn_config['load_params']:
            self.load_rep_fn(cfg.rep_fn_config['path'])
        if 'load_params' in self.cfg.val_fn_config and self.cfg.val_fn_config['load_params']:
            self.load_val_fn(cfg.val_fn_config['path'])

        self.env = cfg.env_fn()
        self.vf_loss = cfg.vf_loss_fn()
        self.constr_fn = cfg.constr_fn()
        self.replay = cfg.replay_fn()

        self.eval_set = self.property_evaluation_dataset(cfg.eval_data)
        self.true_q_predictor = self.cfg.tester_fn.get('true_value_estimator', lambda x:None)

        self.state = None
        self.action = None
        self.next_state = None

    def step(self):
        if self.reset is True:
            self.state = self.env.reset()
            self.reset = False
        action = self.policy(self.state, self.cfg.eps_schedule())
        next_state, reward, done, _ = self.env.step([action])
        self.replay.feed([self.state, action, reward, next_state, int(done)])
        prev_state = self.state
        self.state = next_state
        self.update_stats(reward, done)
        self.update()
        return prev_state, action, reward, next_state, int(done)

    def no_grad_value(self, state):
        with torch.no_grad():
            phi = self.rep_net(self.cfg.state_normalizer(state))
            q_values = self.val_net(phi)
        q_values = torch_utils.to_np(q_values).flatten()
        return q_values

    def policy(self, state, eps):
        q_values = self.no_grad_value(state)
        if self.agent_rng.rand() < eps:
            action = self.agent_rng.randint(0, len(q_values))
        else:
            action = self.agent_rng.choice(np.flatnonzero(q_values == q_values.max()))
        return action

    def update(self):
        if (not self.cfg.rep_fn_config['train_params']) and (not self.cfg.val_fn_config['train_params']):
            return
        
        states, actions, rewards, next_states, terminals = self.replay.sample()
        states = self.cfg.state_normalizer(states)
        next_states = self.cfg.state_normalizer(next_states)

        actions = torch_utils.tensor(actions, self.cfg.device).long()
        if not self.cfg.rep_fn_config['train_params']:
            with torch.no_grad():
                phi = self.rep_net(states)
        else:
            phi = self.rep_net(states)

        if not self.cfg.val_fn_config['train_params']:
            with torch.no_grad():
                q = self.val_net(phi)[self.batch_indices, actions]
        else:
            q = self.val_net(phi)[self.batch_indices, actions]

        # Constructing the target
        with torch.no_grad():
            q_next = self.targets.val_net(self.targets.rep_net(next_states))
            q_next = q_next.max(1)[0]
            terminals = torch_utils.tensor(terminals, self.cfg.device)
            rewards = torch_utils.tensor(rewards, self.cfg.device)
            target = self.cfg.discount * q_next * (1 - terminals).float()
            target.add_(rewards.float())
        
        loss = self.vf_loss(q, target)
        constr = self.constr_fn(phi, q, target)
        
        # if self.independent_constr_opt:
        #     self.constr_optimizer.zero_grad()
        #     constr.backward(retain_graph=True)
        #     self.constr_optimizer.step()
        # else:
        # print("Loss: {}, Contr {}".format(loss, constr))
        loss += constr

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.cfg.tensorboard_logs and self.total_steps % self.cfg.tensorboard_interval == 0:
            self.cfg.logger.tensorboard_writer.add_scalar('dqn/loss/val_loss', loss.item(), self.total_steps)

        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
            self.targets.val_net.load_state_dict(self.val_net.state_dict())

    def eval_step(self, state):
        with torch.no_grad():
            q_values = self.val_net(self.rep_net(self.cfg.state_normalizer(state)))
            q_values = torch_utils.to_np(q_values).flatten()
        return self.agent_rng.choice(np.flatnonzero(q_values == q_values.max()))

    # def log_tensorboard(self):
    #     rewards = self.ep_returns_queue#[: min(self.stats_counter, self.cfg.stats_queue_size)]
    #     mean, median, min_, max_ = np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards)
    #     self.cfg.logger.tensorboard_writer.add_scalar('dqn/reward/average_reward', mean, self.total_steps)
    #     self.cfg.logger.tensorboard_writer.add_scalar('dqn/reward/median_reward', median, self.total_steps)
    #     self.cfg.logger.tensorboard_writer.add_scalar('dqn/reward/min_reward', min_, self.total_steps)
    #     self.cfg.logger.tensorboard_writer.add_scalar('dqn/reward/max_reward', max_, self.total_steps)

    def visualize(self):
        # a plot with rows = num of goal-states and cols = 1
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

        frame = self.state.astype(np.uint8)
        figure, ax = plt.subplots()
        ax.imshow(frame)
        plt.axis('off')

        viz_dir = self.cfg.get_visualization_dir()
        # viz_file = 'visualization_{}.png'.format(self.num_episodes)
        viz_file = 'visualization_{}.png'.format(self.total_steps)
        # viz_file = 'visualization.png'
        plt.savefig(os.path.join(viz_dir, viz_file))
        plt.close()

    def save(self, early=False):
        parameters_dir = self.cfg.get_parameters_dir()
        if early:
            path = os.path.join(parameters_dir, "rep_net_earlystop")
        elif self.cfg.checkpoints:
            path = os.path.join(parameters_dir, "rep_net_{}".format(self.total_steps))
        else:
            path = os.path.join(parameters_dir, "rep_net")
        torch.save(self.rep_net.state_dict(), path)

        if early:
            path = os.path.join(parameters_dir, "val_net_earlystop")
        else:
            path = os.path.join(parameters_dir, "val_net")
        torch.save(self.val_net.state_dict(), path)

    def load_rep_fn(self, parameters_dir):
        path = os.path.join(self.cfg.data_root, parameters_dir)
        self.rep_net.load_state_dict(torch.load(path, map_location=self.cfg.device))
        self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
        self.cfg.logger.info("Load rep function from {}".format(path))

    def load_val_fn(self, parameters_dir):
        path = os.path.join(self.cfg.data_root, parameters_dir)
        self.val_net.load_state_dict(torch.load(path, map_location=self.cfg.device))
        self.targets.val_net.load_state_dict(self.val_net.state_dict())
        self.cfg.logger.info("Load value function from {}".format(path))

    # def log_overestimation(self):
    #     states, actions, true_qs = self.populate_returns(log_traj=True)
    #     states = np.array(states)
    #     true_qs = np.array(true_qs)
    #     with torch.no_grad():
    #         phis = self.rep_net(self.cfg.state_normalizer(states))
    #         q_values = self.val_net(phis)
    #         q_values = torch_utils.to_np(q_values)
    #     onpolicy_q = q_values[np.arange(len(q_values)), actions]
    #     all_diff = onpolicy_q - true_qs
    #     log_str = 'total steps %d, total episodes %3d, ' \
    #               'Overestimation: %.8f/%.8f/%.8f (mean/min/max)'
    #     self.cfg.logger.info(log_str % (self.total_steps, len(self.episode_rewards), all_diff.mean(), all_diff.min(), all_diff.max()))