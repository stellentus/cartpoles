import os

from core.environment.acrobot import Acrobot


class EnvFactory:
    @classmethod
    def create_env_fn(cls, cfg):
        if cfg.env_name == 'Acrobot':
            return lambda: Acrobot(cfg.seed)
        else:
            print(cfg.env_name)
            raise NotImplementedError