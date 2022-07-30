import os

from core.environment.acrobot import Acrobot, CustomizedAcrobot
from core.environment.puddleworld import PuddleWorld


class EnvFactory:
    @classmethod
    def create_env_fn(cls, cfg):
        if cfg.env_name == 'Acrobot':
            return lambda: Acrobot(cfg.seed)
        if cfg.env_name == 'CustomizedAcrobot':
            return lambda: CustomizedAcrobot(cfg.seed)
        if cfg.env_name == 'PuddleWorld':
            return lambda: PuddleWorld(cfg.seed, cfg.env_randomstart)
        else:
            print(cfg.env_name)
            raise NotImplementedError