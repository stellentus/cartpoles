import os
import numpy as np
import torch
import torch.nn as nn

from core.network import network_architectures
from core.utils import torch_utils

class DefaultRepresentation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        net = self.create_network(cfg)
        self.net = net
        self.output_dim = cfg.rep_fn_config['out_dim']

    def forward(self, x):
        return self.net(x)


class NNetRepresentation(DefaultRepresentation):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.activation_config['name'] in ["None", "ReLU"]:
            self.output_dim = cfg.rep_fn_config['out_dim']
        elif cfg.activation_config['name'] == 'FTA':
            self.output_dim = cfg.rep_fn_config['out_dim'] * cfg.activation_config['tile']
        else:
            raise NotImplementedError

    def create_network(self, cfg):
        if cfg.rep_fn_config['network_type'] == 'fc':
            return network_architectures.FCNetwork(cfg.device, np.prod(cfg.rep_fn_config['in_dim']),
                                                   cfg.rep_fn_config['hidden_units'], cfg.rep_fn_config['out_dim'],
                                                   head_activation=cfg.rep_activation_fn, init_type=cfg.rep_fn_config['init_type'])
        elif cfg.rep_fn_config['network_type'] == 'conv':
            return network_architectures.ConvNetwork(cfg.device, cfg.rep_fn_config['in_dim'],
                                                     cfg.rep_fn_config['out_dim'], cfg.rep_fn_config['conv_architecture'],
                                                  head_activation=cfg.rep_activation_fn, init_type=cfg.rep_fn_config['init_type'])
        elif cfg.rep_fn_config['network_type'] is None:
            return lambda x: x
        else:
            raise NotImplementedError


class FlattenRepresentation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.output_dim = cfg.rep_fn_config['out_dim']
        self.device = cfg.device

    def forward(self, x):
        return torch.flatten(torch_utils.tensor(x, self.device), start_dim=1)


class RawSA:
    def __init__(self, cfg):
        self.output_dim = np.prod(cfg.rep_fn_config['in_dim']) * cfg.action_dim
        self.device = cfg.device

    def __call__(self, state, action):
        assert len(state.shape) == 1
        vec = np.zeros(self.output_dim)
        vec[action*len(state): (action+1)*len(state)] = state
        return vec
    
    def parameters(self):
        return []
    
    def state_dict(self):
        return []

    def load_state_dict(self, item):
        return
    
    
class IdentityRepresentation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.output_dim = cfg.rep_fn_config['out_dim']
        self.device = cfg.device

    def forward(self, x):
        return torch_utils.tensor(x, self.device)

    # def parameters(self):
    #     return []
    #
    # def state_dict(self):
    #     return []
    #
    # def load_state_dict(self, item):
    #     return
