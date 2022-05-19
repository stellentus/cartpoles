import os

from core.utils import torch_utils

class EmptyConfig:
    def __init__(self):
        self.exp_name = 'test'
        self.data_root = None
        self.run = 0
        self.param_setting = 0
        self.logger = None
        self.log_observations = False
        self.tensorboard_logs = False
        self.batch_size = 0
        self.replay_with_len = False
        self.memory_size = 1
        self.evaluation_criteria = "return"
        self.checkpoints = False
        self.warm_up_step = 0
        return

    def get_log_dir(self):
        d = os.path.join(self.data_root, self.exp_name, "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting))
        torch_utils.ensure_dir(d)
        return d

    def log_config(self):
        attrs = self.get_print_attrs()
        for param, value in attrs.items():
            self.logger.info('{}: {}'.format(param, value))

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        return attrs

    @property
    def env_fn(self):
        return self.__env_fn

    @env_fn.setter
    def env_fn(self, env_fn):
        self.__env_fn = env_fn
        self.state_dim = env_fn().state_dim
        self.action_dim = env_fn().action_dim

    def get_parameters_dir(self):
        d = os.path.join(self.data_root, self.exp_name,
                         "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting),
                         "parameters")
        torch_utils.ensure_dir(d)
        return d

class Config(EmptyConfig):
    def __init__(self):
        super().__init__()
        self.exp_name = 'test'
        self.data_root = None
        self.device = None
        self.run = 0
        self.param_setting = 0

        self.env_name = None
        self.state_dim = None
        self.action_dim = None
        self.max_steps = 0

        self.log_interval = int(1e3)
        self.save_interval = 0
        self.eval_interval = 0
        self.num_eval_episodes = 5
        self.timeout = None
        self.stats_queue_size = 10

        self.__env_fn = None
        self.logger = None

        self.tensorboard_logs = False
        self.tensorboard_interval = 100

        self.state_normalizer = None
        self.state_norm_coef = 0
        self.reward_normalizer = None
        self.reward_norm_coef = 1.0

        self.early_cut_off = False

        self.testset_paths = None
        self.tester_fn_config = {}
        self.evalset_path = {}
        self.true_value_paths = None
        self.visualize = False
        self.evaluate_overestimation = False

    def get_log_dir(self):
        d = os.path.join(self.data_root, self.exp_name, "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting))
        torch_utils.ensure_dir(d)
        return d

    def log_config(self):
        attrs = self.get_print_attrs()
        for param, value in attrs.items():
            self.logger.info('{}: {}'.format(param, value))

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        return attrs

    @property
    def env_fn(self):
        return self.__env_fn

    @env_fn.setter
    def env_fn(self, env_fn):
        self.__env_fn = env_fn
        self.state_dim = env_fn().state_dim
        self.action_dim = env_fn().action_dim

    def get_visualization_dir(self):
        d = os.path.join(self.data_root, self.exp_name, "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting), "visualizations")
        torch_utils.ensure_dir(d)
        return d

    def get_parameters_dir(self):
        d = os.path.join(self.data_root, self.exp_name,
                         "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting),
                         "parameters")
        torch_utils.ensure_dir(d)
        return d

    def get_data_dir(self):
        d = os.path.join(self.data_root, self.exp_name,
                         "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting),
                         "data")
        torch_utils.ensure_dir(d)
        return d

    def get_warmstart_property_dir(self):
        d = os.path.join(self.data_root, self.exp_name,
                         "{}_run".format(self.run),
                         "{}_param_setting".format(self.param_setting),
                         "warmstart_property")
        torch_utils.ensure_dir(d)
        return d

    def get_logdir_format(self):
        return os.path.join(self.data_root, self.exp_name,
                            "{}_run",
                            "{}_param_setting".format(self.param_setting))


class DQNAgentConfig(Config):
    def __init__(self):
        super().__init__()
        self.agent = 'DQNAgent'
        self.learning_rate = 0

        self.decay_epsilon = False
        self.epsilon = 0.1
        self.epsilon_start = None
        self.epsilon_end = None
        self.eps_schedule = None
        self.epsilon_schedule_steps = None
        self.random_action_prob = None

        self.discount = None

        self.network_type = 'fc'
        self.batch_size = None
        self.use_target_network = True
        self.memory_size = None
        self.optimizer_type = 'Adam'
        self.optimizer_fn = None
        self.val_net = None
        self.target_network_update_freq = None
        self.update_network = True

        self.replay = True
        self.replay_fn = None
        self.replay_with_len = False
        self.memory_size = 10000

        self.evaluation_criteria = "return"
        self.vf_loss = "mse"
        self.vf_loss_fn = None
        self.constraint = None
        self.constr_fn = None
        self.constr_weight = 0
        self.rep_type = "default"

        self.save_params = False
        self.save_early = None

        self.activation_config = {'name': 'None'}

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'eps_schedule', 'optimizer_fn', 'constr_fn',
                  'replay_fn', 'vf_loss_fn',
                  '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'rep_fn', 'val_fn', 'rep_activation_fn']:
            del attrs[k]
        return attrs


class SarsaAgentConfig(Config):
    def __init__(self):
        super().__init__()
        self.agent = 'SarsaAgent'
        self.learning_rate = 0
        self.discount = None
        
        self.network_type = 'fc'
        self.batch_size = 1
        self.use_target_network = False
        self.target_network_update_freq = None
        self.memory_size = None
        self.optimizer_type = 'Adam'
        self.optimizer_fn = None
        self.val_net = None
        self.update_network = True

        self.replay = False

        self.evaluation_criteria = "return"
        self.vf_loss = "mse"
        self.vf_loss_fn = None
        self.constraint = None
        self.constr_fn = None
        self.constr_weight = 0
        self.rep_type = "default"

        self.save_params = False
        self.save_early = None

        self.activation_config = {'name': 'None'}

        self.decay_epsilon = False
        self.epsilon = 0.1
        self.epsilon_start = None
        self.epsilon_end = None
        self.eps_schedule = None
        self.epsilon_schedule_steps = None
        self.random_action_prob = None

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'eps_schedule', 'optimizer_fn', 'constr_fn',
                  'replay_fn', 'vf_loss_fn',
                  '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'rep_fn', 'val_fn', 'rep_activation_fn']:
            del attrs[k]
        return attrs


class SarsaOfflineConfig(SarsaAgentConfig):
    def __init__(self):
        super().__init__()
        self.agent = 'SarsaOffline'
        self.learning_rate = 0
        self.discount = None
        self.early_cut_threshold = 3
        self.tester_fn_config = []

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'eps_schedule', 'optimizer_fn', 'constr_fn',
                  'replay_fn', 'vf_loss_fn',
                  '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'rep_fn', 'val_fn', 'rep_activation_fn',
                  'offline_data', 'tester_fn', 'eval_data']:
            del attrs[k]
        return attrs
    
    
class SarsaOfflineBatchConfig(SarsaOfflineConfig):
    def __init__(self):
        super().__init__()
        self.agent = 'SarsaOfflineBatch'
        
        
class MonteCarloOfflineConfig(Config):
    def __init__(self):
        super().__init__()
        self.agent = 'MonteCarloOffline'
        self.activation_config = {'name': 'None'}
        self.constraint = None
        self.vf_loss = "mse"
        self.vf_loss_fn = None
        self.batch_size = 32
        self.early_cut_threshold = 3
        self.tester_fn_config = []

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'eps_schedule', 'optimizer_fn', 'constr_fn',
                  'replay_fn', 'vf_loss_fn',
                  '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'rep_fn', 'val_fn', 'rep_activation_fn',
                  'offline_data', 'tester_fn', 'eval_data']:
            del attrs[k]
        return attrs


class MonteCarloConfig(Config):
    def __init__(self):
        super().__init__()
        self.agent = 'MonteCarloAgent'
        self.activation_config = {'name': 'None'}
        self.constraint = None
        self.vf_loss = "mse"

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'eps_schedule', 'optimizer_fn', 'constr_fn',
                  'replay_fn', 'vf_loss_fn',
                  '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'rep_fn', 'val_fn', 'rep_activation_fn']:
            del attrs[k]
        return attrs

class CQLOfflineConfig(MonteCarloOfflineConfig):
    def __init__(self):
        super().__init__()
        self.agent = 'CQLAgentOffline'
        self.cql_alpha = 1.0


class FQIConfig(DQNAgentConfig):
    def __init__(self):
        super().__init__()
        self.agent = 'FQI'
        self.early_cut_threshold = 3
        self.tester_fn_config = []
        self.memory_size = 0

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'eps_schedule', 'optimizer_fn', 'constr_fn',
                  'replay_fn', 'vf_loss_fn',
                  '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'rep_fn', 'val_fn', 'rep_activation_fn',
                  'offline_data', 'tester_fn', 'eval_data']:
            del attrs[k]
        return attrs

class QmaxCloneConfig(Config):
    def __init__(self):
        super().__init__()
        self.agent = 'QmaxCloneAgent'
        self.learning_rate = 0

        self.decay_epsilon = False
        self.epsilon = 0
        self.epsilon_start = None
        self.epsilon_end = None
        self.eps_schedule = None
        self.epsilon_schedule_steps = None
        self.random_action_prob = None

        self.discount = None

        self.network_type = 'fc'
        self.batch_size = None
        self.use_target_network = False
        self.memory_size = None
        self.optimizer_type = 'Adam'
        self.optimizer_fn = None
        self.val_net = None

        self.replay = False
        # self.replay_fn = None
        # self.replay_with_len = False
        # self.memory_size = 10000

        self.evaluation_criteria = "return"
        self.vf_loss = "mse"
        self.vf_loss_fn = None
        self.constraint = None
        self.constr_fn = None
        self.constr_weight = 0
        self.rep_type = "default"

        self.save_params = False
        self.save_early = None

        self.activation_config = {'name': 'None'}

        self.early_cut_threshold = 3
        self.tester_fn_config = []
        self.memory_size = 0

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['logger', 'eps_schedule', 'optimizer_fn', 'constr_fn',
                  'replay_fn', 'vf_loss_fn',
                  '_Config__env_fn', 'state_normalizer',
                  'reward_normalizer', 'rep_fn', 'val_fn', 'rep_activation_fn',
                  'offline_data', 'tester_fn', 'eval_data']:
            del attrs[k]
        return attrs


class QmaxConstrConfig(QmaxCloneConfig):
    def __init__(self):
        super().__init__()
        self.agent = 'QmaxConstrAgent'
        self.constr_weight = 1.0

