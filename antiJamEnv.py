#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
from gym import spaces
import numpy as np


class AntiJamEnv(gym.Env):
    def __init__(self, jammer_type, channel_switching_cost):
        super(AntiJamEnv, self).__init__()

        self.observation = None
        self.jammer_frequency = None
        self.jammer_mode = None
        self.current_channel = None
        self.num_channels = 8
        self.channel_bandwidth = 20  # MHz
        self.frequency_range = [5180, 5320]  # MHz
        self.frequency_lists = range(5180, 5340, 20)  # MHz

        self.observation_space = spaces.Box(low=-30, high=40, shape=(self.num_channels,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_channels)
        self.jammer_type = jammer_type
        self.jammer_modes = ['constant', 'random', 'sweeping']
        self.csc = channel_switching_cost

        self._max_episode_steps = None

    def reset(self):
        self.current_channel = np.random.randint(self.num_channels)

        if self.jammer_type == 'dynamic':
            self.jammer_mode = np.random.choice(self.jammer_modes)
        else:
            self.jammer_mode = self.jammer_type
        self.jammer_frequency = self.frequency_lists[self.current_channel]

        self.observation = np.array([self._get_received_power(i) for i in range(self.num_channels)])
        return self.observation

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        received_power = self._get_received_power(action)
        if received_power >= 0:
            reward = 0.0
        else:
            reward = 1.0

        if self.current_channel != action:
            reward -= self.csc  # Channel switching cost

        self.current_channel = action

        if self.jammer_mode == 'random':
            self.jammer_frequency = np.random.uniform(self.frequency_range[0], self.frequency_range[1])
        elif self.jammer_mode == 'sweeping':
            self.jammer_frequency += self.channel_bandwidth
            if self.jammer_frequency > self.frequency_range[1]:
                self.jammer_frequency = self.frequency_range[0]

        self.observation = np.array([self._get_received_power(i) for i in range(self.num_channels)])

        return self.observation, reward, False, {}

    def _get_received_power(self, channel_idx):
        # Simulate received jamming power using normal distribution
        jammed_power = np.random.normal(loc=30, scale=5)
        adjacent_power = np.random.normal(loc=13, scale=3)
        far_away_power = np.random.normal(loc=-7, scale=1)

        if channel_idx == self.current_channel:
            return jammed_power
        elif abs(channel_idx - self.current_channel) == 1:
            return adjacent_power
        else:
            return far_away_power

    def render(self, mode='human'):
        pass

    def close(self):
        pass