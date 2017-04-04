'''
GymImageMDPClass.py: Contains implementation for MDPs of the Gym Environments where observations are images.
'''

# Python imports.
import numpy as np
import random
import sys
import os
import random

# Local imports.
try:
    import gym
except ImportError:
    print "Error: you do not have gym installed. See https://github.com/openai/gym."
    quit()

try:
    import cv2
except ImportError:
    print "Error: you do not have OpenCV installed. See https://pypi.python.org/pypi/opencv-python."
    quit()

from ...mdp.MDPClass import MDP
from GymStateClass import GymState


class GymImageMDP(MDP):
    ''' Class for Gym MDPs with image observations only'''

    def __init__(self, env_name='Breakout-v0', frame_width=84, frame_height=84, frame_skip=4, history_size=4, max_noop_acts=30, render=False, use_luminance=True):
        '''
        Args:
            env_name (str)
        '''
        self.env_name = env_name
        self.env = gym.make(env_name)

        self.width = frame_width
        self.height = frame_height
        self.frame_skip = frame_skip
        self.history = history_size
        self.noop_max = max_noop_acts
        self.frame_history = [np.zeros((self.width, self.height)) * self.history]

        if self.env.__dict__.get('frameskip', None):
            self.env.__dict__['frameskip'] = (self.frame_skip, self.frame_skip+1)

        for _ in range(np.random.randint(self.noop_max+1)):
            self.env.step(0)

        if render:
            self.env.render()

        if use_luminance:
            self.preprocess = lambda x: self.compute_luminance(x)
        else:
            self.preprocess = lambda x: x

        MDP.__init__(self, xrange(self.env.action_space.n), self._transition_func, self._reward_func,
                     init_state=GymState(self.build_next_state(self.env.reset())))

    def _reward_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (float)
        '''
        obs, reward, is_terminal, info = self.env.step(action)

        self.next_state = GymState(self.build_next_state(obs), is_terminal=is_terminal)

        return reward

    def _transition_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (State)
        '''
        return self.next_state

    def compute_luminance(self, img):
        channels = np.dsplit(img, 3)
        lum = np.array(0.2126*channels[0] + 0.7152*channels[1] + 0.0722*channels[2])
        lum = lum.reshape([self.width, self.height, 1])
        lum = cv2.resize(lum, (self.width, self.height))
        return lum

    def build_next_state(self, img):
        preprocessed = self.preprocess(img)
        self.frame_history[:-1] = self.frame_history[1:]
        self.frame_history[-1] = preprocessed
        return np.dstack(self.frame_history)

    def reset(self):
        self.env.reset()

        for _ in range(np.random.randint(self.noop_max+1)):
            self.env.step(0)

    def __str__(self):
        return "gym-" + str(self.env_name)

