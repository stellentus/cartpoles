import math

import numpy as np
import gym
import copy

import core.utils.helpers
from core.utils.torch_utils import random_seed


class Acrobot:
    def __init__(self, seed=np.random.randint(int(1e5))):
        random_seed(seed)
        self.state_dim = (6,)
        self.action_dim = 3
        self.env = gym.make('Acrobot-v1')
        self.env._seed = seed
        self.env._max_episode_steps = np.inf # control timeout setting in agent
        self.state = None

    def generate_state(self, coords):
        return coords

    def reset(self):
        self.state = np.asarray(self.env.reset())
        return self.state

    def step(self, a):
        # To fit the go lang implementation
        # go implementation: 2 actions in total, if act=0 then torque=+1, if act=1 then torque=-1.
        act = 2 if a[0] == 0 else 0

        state, reward, done, info = self.env.step(act)
        self.state = state
        reward = -1 # always -1
        # self.env.render()
        return np.asarray(state), np.asarray(reward), np.asarray(done), info

    def get_visualization_segment(self):
        raise NotImplementedError

    def get_useful(self, state=None):
        if state:
            return state
        else:
            return np.array(self.env.state)

    def info(self, key):
        return

    def hack_step(self, current_s, action):
        env = copy.deepcopy(self.env).unwrapped
        env.reset()
        # acos0 = np.arccos(current_s[0])
        # asin0 = np.arcsin(current_s[1])
        # acos1 = np.arccos(current_s[2])
        # asin1 = np.arcsin(current_s[3])
        
        env.state = np.array([core.utils.helpers.arcradians(current_s[0], current_s[1]),
                              core.utils.helpers.arcradians(current_s[2], current_s[3]),
                              current_s[4],
                              current_s[5]])
        # print(env.state)
        state, reward, done, info = env.step(action)
        # print(env.state)
        # print(np.sin(env.state[0]), np.cos(env.state[0]), np.sin(env.state[1]), np.cos(env.state[1]))
        # print(state)
        # print()
        return np.asarray(state), np.asarray(reward), np.asarray(done), info


class CustomizedAcrobot(Acrobot):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super().__init__(seed)
        self.env.unwrapped.LINK_LENGTH_1 *= 2
        self.env.unwrapped.LINK_MASS_1 *= 2
