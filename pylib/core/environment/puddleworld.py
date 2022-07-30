import numpy as np
import copy

GOALTHRESHOLD = 0.1
ACTIONTHRUST  = 0.05
PUDDLEWIDTH   = 0.1

class PuddleWorld:
    def __init__(self, seed=1e-5, env_randomstart=True, normalized=False):
        self.env_rng = np.random.RandomState(seed)
        self.state_dim = (2,)
        self.action_dim = 4

        self.start_state = np.array([0.0, 0.0])
        self.goal_state = np.array([1.0, 1.0])
        self.puddle_centers = np.array([[0.1, 0.75], [0.45, 0.75], [0.45, 0.4], [0.45, 0.8]])
        self.delays = 0
        self.percent_noise = []
        self.start_hard = False
        self.state = np.zeros(2)
        self.buffer = 0
        self.bufferInsertIndex = []
        self.actions = np.zeros((4, 2))

    def initialize_with_settings(self):
        for i in range(4):
            self.actions[i, i//2] = ACTIONTHRUST * (i % 2 * 2 - 1)

    def noisy_state(self, start_s):
        state_lower_bound = np.array([0.0, 0.0])
        state_upper_bound = np.array([1.0, 1.0])
        state = copy.deepcopy(self.state)
        if len(self.percent_noise) != 0:
            for i in range(len(state)):
                state[i] += self.env_rng.random() * self.percent_noise[i]
                state[i] = self.clamp(state[i], state_lower_bound[i], state_upper_bound[i])
        return state

    def clamp(self, x, min, max):
        return np.max([min, np.min([x, max])])

    def randomize_state(self):
        if self.start_hard:
            rnd_x = self.env_rng.random() * (0.35 - 0.3) + 0.3
            rnd_y = self.env_rng.random() * (0.65 - 0.6) + 0.6
            self.start_state = np.array([rnd_x, rnd_y])
            self.state = self.start_state
            return
        while True:
            for i in range(len(self.state)):
                self.start_state[i] = self.env_rng.random()
            if np.linalg.norm(self.state - self.goal_state, ord=1) > GOALTHRESHOLD:
                self.state = self.start_state
                # print(self.state)
                return

    def rand_float(self, min, max, start_s):
        return self.env_rng.random() * (max - min) + min

    def reset(self):
        self.randomize_state()
        return self.state

    def step(self, a):
        # print(a)
        # print(self.actions[a[0]])
        for i in range(len(self.state)):
            self.state[i] += self.actions[a[0], i] + self.env_rng.random() * 0.01
            self.state[i] = self.clamp(self.state[i], 0, 1)
        obs = copy.deepcopy(self.state)#self.get_observations(False)
        reward = self.get_rewards()
        done = np.linalg.norm(self.state - self.goal_state, ord=1) < GOALTHRESHOLD
        return obs, reward, done, ""

    def get_observations(self, start_s):
        obs = self.noisy_state(False)
        return obs

    def get_rewards(self):
        reward = -1.0

        dist = 0
        if self.state[0] < self.puddle_centers[0, 0] and np.linalg.norm(self.state - self.puddle_centers[0]) < PUDDLEWIDTH:
            # left semicircle
            dist = PUDDLEWIDTH - np.linalg.norm(self.state - self.puddle_centers[0])
        elif self.in_range(self.state[0], self.puddle_centers[0, 0], self.puddle_centers[1, 0] - PUDDLEWIDTH) and \
            self.in_range(self.state[1], self.puddle_centers[0, 1] - PUDDLEWIDTH, self.puddle_centers[0,1]+PUDDLEWIDTH):
            # left rectangle
            dist = PUDDLEWIDTH - np.abs(self.state[1] - self.puddle_centers[0,1])
        elif self.state[1] > self.puddle_centers[1,1]+PUDDLEWIDTH and np.linalg.norm(self.state - self.puddle_centers[3]) < PUDDLEWIDTH:
            # top right semicircle
            dist = PUDDLEWIDTH - np.linalg.norm(self.state - self.puddle_centers[3])
        elif self.state[1] < self.puddle_centers[2,1] and np.linalg.norm(self.state - self.puddle_centers[2]) < PUDDLEWIDTH:
            # bottom right semicircle
            dist = PUDDLEWIDTH - np.linalg.norm(self.state - self.puddle_centers[2])
        elif self.in_range(self.state[0], self.puddle_centers[2,0]-PUDDLEWIDTH, self.puddle_centers[2,0]+PUDDLEWIDTH):
            if self.in_range(self.state[1], self.puddle_centers[2,1], self.puddle_centers[1,1]-PUDDLEWIDTH):
                # bottom right rectangle
                dist = PUDDLEWIDTH - np.abs(self.state[0] - self.puddle_centers[2,0])
            elif self.in_range(self.state[1], self.puddle_centers[1,1]-PUDDLEWIDTH, self.puddle_centers[1,1]+PUDDLEWIDTH):
                # top right rectangle
                if self.state[1] >= self.puddle_centers[3,1]:
                    # the dist is either to the arc or the short edge
                    if self.state[0] < (9 - np.sqrt(3))/20.0 :
                        # small rectangle
                        dist = self.puddle_centers[1,1] + PUDDLEWIDTH - self.state[1]
                    elif self.state[0]*0.57735+self.state[1]<1.05981:
                        dist = np.linalg.norm(self.state - np.array([(9.0 - np.sqrt(3)) / 20.0, self.puddle_centers[1,1]]))
                    else:
                        dist = PUDDLEWIDTH - np.linalg.norm(self.state - self.puddle_centers[3])
                else:
                    if self.state[0] < (9.0 - np.sqrt(3))/20.0:
                        dist_to_bottom_left_corner = np.linalg.norm(self.state -
                                                                    np.array([self.puddle_centers[1,0] - PUDDLEWIDTH, self.puddle_centers[1,1] - PUDDLEWIDTH]))
                        dist_to_top_edge = self.puddle_centers[1,1] + PUDDLEWIDTH - self.state[1]
                        dist = np.min([dist_to_bottom_left_corner, dist_to_top_edge])
                    else:
                        dist_to_top_left_corner = np.linalg.norm(self.state -
                                                                 np.array([(9.0 - np.sqrt(3))/20.0, self.puddle_centers[1,1]+PUDDLEWIDTH]))
                        dist_to_bottom_left_corner = np.linalg.norm(self.state -
                                                                    np.array([self.puddle_centers[1,0] - PUDDLEWIDTH, self.puddle_centers[1,1]-PUDDLEWIDTH]))
                        dist_to_right_side = self.puddle_centers[1,0] + PUDDLEWIDTH - self.state[0]
                        dist = np.min([dist_to_top_left_corner, dist_to_bottom_left_corner, dist_to_right_side])
        reward -= 400.0 * dist
        return reward

    def in_range(self, x, min, max):
        return x >= min and x <= max

