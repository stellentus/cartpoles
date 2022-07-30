import os

from core.agent.cql import *
from core.agent.dqn import *


class AgentFactory:
    @classmethod
    def create_agent_fn(cls, cfg):
        if cfg.agent_name == 'CQLAgentOffline':
            return lambda: CQLAgentOffline(cfg)
        elif cfg.agent_name == 'DQNAgent':
            return lambda: DQNAgent(cfg)
        else:
            print(cfg.agent_name)
            raise NotImplementedError