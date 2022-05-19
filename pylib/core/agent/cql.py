import os
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import torch

from core.agent import base
from core.utils import torch_utils


class CQLAgentOffline(base.Agent):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.rep_net = cfg.rep_fn()
        self.val_net = cfg.val_fn()

        # Creating target networks for value, representation, and auxiliary val_net
        rep_net_target = cfg.rep_fn()
        rep_net_target.load_state_dict(self.rep_net.state_dict())
        val_net_target = cfg.val_fn()
        val_net_target.load_state_dict(self.val_net.state_dict())

        # Creating Target Networks
        TargetNets = namedtuple('TargetNets', ['rep_net', 'val_net'])
        self.targets = TargetNets(rep_net=rep_net_target, val_net=val_net_target)

        params = list(self.rep_net.parameters()) + list(self.val_net.parameters())
        self.optimizer = cfg.optimizer_fn(params)

        if 'load_params' in self.cfg.rep_fn_config and self.cfg.rep_fn_config['load_params']:
            self.load_rep_fn(cfg.rep_fn_config['path'])
        if 'load_params' in self.cfg.val_fn_config and self.cfg.val_fn_config['load_params']:
            self.load_val_fn(cfg.val_fn_config['path'])

        self.env = cfg.env_fn()
        self.vf_loss = cfg.vf_loss_fn()
        self.constr_fn = cfg.constr_fn()

        self.alpha = cfg.cql_alpha

        self.prev_state = None
        self.prev_action = None
        self.state = None
        self.reward = None
        self.done = None

        self.true_q_predictor = self.cfg.tester_fn.get('true_value_estimator', lambda x: None)
        self.trainset, self.testset = self.training_set_construction(cfg.offline_data)
        self.training_size = len(self.trainset[0])
        self.training_indexs = np.arange(self.training_size)
        self.eval_set = self.property_evaluation_dataset(cfg.eval_data)
        
        self.training_loss = []
        self.test_loss = []
        self.tloss_increase = 0
        self.tloss_rec = np.inf
    
    def step(self):
        train_s, train_a, train_r, train_ns, train_t, train_na, _, _, _ = self.trainset
        
        self.agent_rng.shuffle(self.training_indexs)
        ls_epoch = []
        for b in range(int(np.ceil(self.training_size / self.cfg.batch_size))):
            idxs = self.training_indexs[b * self.cfg.batch_size: (b + 1) * self.cfg.batch_size]
            in_ = torch_utils.tensor(self.cfg.state_normalizer(train_s[idxs]), self.cfg.device)
            act = train_a[idxs]
            r = torch_utils.tensor(train_r[idxs], self.cfg.device)
            ns = torch_utils.tensor(self.cfg.state_normalizer(train_ns[idxs]), self.cfg.device)
            t = torch_utils.tensor(train_t[idxs], self.cfg.device)
            na = train_na[idxs]

            """
            According to https://github.com/BY571/CQL/blob/main/CQL-DQN/agent.py
            def learn(self, experiences):
            """
            q_s = self.val_net(self.rep_net(in_))
            q_s_a = q_s[np.arange(len(in_)), act]
            q_pred = q_s_a
            with torch.no_grad():
                q_tar = r + (self.cfg.discount * (1 - t) * self.targets.val_net(self.targets.rep_net(ns)).max(1)[0])
            loss = self.cql_loss(q_s, q_s_a, q_pred, q_tar)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.rep_net.parameters()) + list(self.val_net.parameters()), 1)
            self.optimizer.step()
            ls_epoch.append(torch_utils.to_np(loss))

        
        self.training_loss.append(np.array(ls_epoch).mean())
        self.update_stats(0, None)
        if self.cfg.use_target_network and self.total_steps % self.cfg.target_network_update_freq == 0:
            self.targets.rep_net.load_state_dict(self.rep_net.state_dict())
            self.targets.val_net.load_state_dict(self.val_net.state_dict())
        
        return self.test_fn()
    
    def cql_loss(self, q_s, q_s_a, q_pred, q_tar):
        cql1_loss = torch.logsumexp(q_s, dim=1).mean() - q_s_a.mean()
        bellmann_error = self.vf_loss(q_pred, q_tar)
        loss = self.alpha * cql1_loss + 0.5 * bellmann_error
        return loss

    def test_fn(self):
        test_s, test_a, test_r, test_sp, test_term, _, _, _, _ = self.testset  # test_ap will be replaced if following the current estimation
        test_s = torch_utils.tensor(self.cfg.state_normalizer(test_s), self.cfg.device)
        test_r = torch_utils.tensor(test_r, self.cfg.device)
        test_sp = torch_utils.tensor(self.cfg.state_normalizer(test_sp), self.cfg.device)
        test_term = torch_utils.tensor(test_term, self.cfg.device)
        with torch.no_grad():
            q_s = self.val_net(self.rep_net(test_s))
            q_s_a = q_s[np.arange(len(test_s)), test_a]
            q_pred = q_s_a
            q_tar = test_r + (self.cfg.discount * (1 - test_term) * self.targets.val_net(self.targets.rep_net(test_sp)).max(1)[0])
            tloss = self.cql_loss(q_s, q_s_a, q_pred, q_tar)

        if tloss - self.tloss_rec > 0:
            self.tloss_increase += 1
        else:
            self.tloss_increase = 0
        self.tloss_rec = tloss
        self.test_loss.append(tloss)
        if self.tloss_increase > self.cfg.early_cut_threshold:
            return "EarlyCutOff"
        return
    
    def log_file(self, elapsed_time=-1):
        if len(self.training_loss) > 0:
            training_loss = np.array(self.training_loss)
            self.training_loss = []
            mean, median, min_, max_ = np.mean(training_loss), np.median(training_loss), np.min(training_loss), np.max(training_loss)
            
            test_loss = np.array(self.test_loss)
            self.test_loss = []
            tmean, tmedian, tmin_, tmax_ = np.mean(test_loss), np.median(test_loss), np.min(test_loss), np.max(test_loss)
            
            log_str = 'TRAIN LOG: epoch %d, ' \
                      'training loss %.4f/%.4f/%.4f/%.4f/%d (mean/median/min/max/num), ' \
                      'test loss %.4f/%.4f/%.4f/%.4f/%d (mean/median/min/max/num), %.4f steps/s'
            self.cfg.logger.info(log_str % (self.total_steps,
                                            mean, median, min_, max_, len(training_loss),
                                            tmean, tmedian, tmin_, tmax_, len(test_loss),
                                            elapsed_time))

            self.populate_returns()
            mean, median, min_, max_ = self.log_return(self.ep_returns_queue_test, "TEST", elapsed_time)
            return tmean, tmedian, tmin_, tmax_
        else:
            log_str = 'TRAIN LOG: epoch %d, ' \
                      'training loss %.2f/%.2f/%.2f/%.2f/%d (mean/median/min/max/num), ' \
                      'test loss %.2f/%.2f/%.2f/%.2f/%d (mean/median/min/max/num), %.2f steps/s'
            self.cfg.logger.info(log_str % (self.total_steps,
                                            np.nan, np.nan, np.nan, np.nan, 0,
                                            np.nan, np.nan, np.nan, np.nan, 0,
                                            elapsed_time))

            self.populate_returns()
            mean, median, min_, max_ = self.log_return(self.ep_returns_queue_test, "TEST", elapsed_time)
            return [None] * 4

    # If there's an online version, move the following to parent class
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

    def eval_step(self, state):
        with torch.no_grad():
            q_values = self.val_net(self.rep_net(torch_utils.tensor(self.cfg.state_normalizer(state), self.cfg.device)))
            q_values = torch_utils.to_np(q_values).flatten()
        return self.agent_rng.choice(np.flatnonzero(q_values == q_values.max()))
