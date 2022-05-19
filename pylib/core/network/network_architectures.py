import numpy as np
import torch
import torch.nn as nn

from core.network import network_utils, network_bodies
from core.utils import torch_utils


class LinearNetwork(nn.Module):
    def __init__(self, device, input_units, output_units, init_type='uniform', bias=True):
        super().__init__()

        if init_type == 'xavier':
            self.fc_head = network_utils.layer_init_xavier(nn.Linear(input_units, output_units, bias=bias))
        elif init_type == 'uniform':
            self.fc_head = network_utils.layer_init_uniform(nn.Linear(input_units, output_units, bias=bias), bias=bias)
        else:
            raise ValueError('init_type is not defined: {}'.format(init_type))

        self.to(device)
        self.device = device

    def forward(self, x):
        if not isinstance(x, torch.Tensor): x = torch_utils.tensor(x, self.device)
        if len(x.shape) > 2: x = x.view(x.shape[0], -1)
        y = self.fc_head(x)
        return y

    def compute_lipschitz_upper(self):
        return [np.linalg.norm(self.fc_head.weight.detach().cpu().numpy(), ord=2)]


class FCNetwork(nn.Module):
    def __init__(self, device, input_units, hidden_units, output_units, head_activation=lambda x:x, init_type='xavier'):
        super().__init__()
        body = network_bodies.FCBody(device, input_units, hidden_units=tuple(hidden_units), init_type=init_type)
        self.body = body
        if init_type == "xavier":
            self.fc_head = network_utils.layer_init_xavier(nn.Linear(body.feature_dim, output_units))
        elif init_type == "uniform":
            self.fc_head = network_utils.layer_init_uniform(nn.Linear(body.feature_dim, output_units))
        else:
            raise ValueError('init_type is not defined: {}'.format(init_type))

        self.device = device
        self.head_activation = head_activation
        self.to(device)

    def forward(self, x):
        if not isinstance(x, torch.Tensor): x = torch_utils.tensor(x, self.device)
        if len(x.shape) > 2: x = x.view(x.shape[0], -1)
        y = self.body(x)
        y = self.fc_head(y)
        y = self.head_activation(y)
        return y

    def compute_lipschitz_upper(self):
        lips = self.body.compute_lipschitz_upper()
        lips.append(np.linalg.norm(self.fc_head.weight.detach().cpu().numpy(), ord=2))
        return lips


class Constant(nn.Module):
    def __init__(self, device, out_dim, constant):
        super().__init__()
        self.device = device
        self.constant = torch_utils.tensor([constant]*out_dim, self.device)
    
    def __call__(self, *args, **kwargs):
        return self.constant