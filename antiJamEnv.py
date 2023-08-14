#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import tensorflow as tf
import tf_slim as slim
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
from tensorflow import keras
from ns3gym import ns3env
import gym
from gym import spaces
import numpy as np


class AntiJamEnv(gym.Env):
    def __init__(self):
        super(AntiJamEnv, self).__init__()

        self.num_channels = 8
        self.channel_bandwidth = 20  # MHz
        self.frequency_range = [5180, 5320]  # MHz

        self.observation_space = spaces.Box(low=-30, high=40, shape=(self.num_channels,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_channels)

        self.current_channel = np.random.randint(self.num_channels)
        self.jammer_modes = ['constant', 'random', 'sweeping']
        self.jammer_mode = np.random.choice(self.jammer_modes)
        self.jammer_frequency = np.random.uniform(self.frequency_range[0], self.frequency_range[1])

    def _get_received_power(self, channel_idx):
        # Simulate received jamming power using normal distribution
        jammed_power = np.random.normal(loc=30, scale=5)
        adjacent_power = np.random.normal(loc=13, scale=3)
        far_away_power = np.random.normal(loc=-7, scale=1)

        if channel_idx == self.current_channel:
            return jammed_power
        elif abs(channel_idx - self.current_channel) == 1:
            return adjacent_power
        elif abs(channel_idx - self.current_channel) >= 3:
            return far_away_power
        else:
            return -30  # Unjammed

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        received_power = self._get_received_power(action)
        if received_power >= 0:
            reward = 1.0
        else:
            reward = 0.0

        if self.current_channel != action:
            reward *= 0.9  # Channel switching cost

        self.current_channel = action

        if self.jammer_mode == 'random':
            self.jammer_frequency = np.random.uniform(self.frequency_range[0], self.frequency_range[1])
        elif self.jammer_mode == 'sweeping':
            self.jammer_frequency += self.channel_bandwidth
            if self.jammer_frequency > self.frequency_range[1]:
                self.jammer_frequency = self.frequency_range[0]

        self.observation = np.array([self._get_received_power(i) for i in range(self.num_channels)])

        return self.observation, reward, False, {}

    def reset(self):
        self.current_channel = np.random.randint(self.num_channels)
        self.jammer_mode = np.random.choice(self.jammer_modes)
        self.jammer_frequency = np.random.uniform(self.frequency_range[0], self.frequency_range[1])

        self.observation = np.array([self._get_received_power(i) for i in range(self.num_channels)])
        return self.observation

    def render(self, mode='human'):
        pass

    def close(self):
        pass


# Test the environment
env = AntiJamEnv()
observation = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)
    print("Action:", action, "Reward:", reward, "Observation:", observation)
    if done:
        break
