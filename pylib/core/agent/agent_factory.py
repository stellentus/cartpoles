import os

from core.agent.cql import *


class AgentFactory:
    @classmethod
    def create_agent_fn(cls, cfg):
        if cfg.agent_name == 'CQLAgentOffline':
            return lambda: CQLAgentOffline(cfg)
        else:
            print(cfg.agent_name)
            raise NotImplementedError