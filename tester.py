#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import json
import streamlit as st
from DDQN import DoubleDeepQNetwork
from antiJamEnv import AntiJamEnv


def test(agent, jammer_type, channel_switching_cost):
    env = AntiJamEnv(jammer_type, channel_switching_cost)
    ob_space = env.observation_space
    ac_space = env.action_space

    s_size = ob_space.shape[0]
    a_size = ac_space.n
    max_env_steps = 3
    TEST_Episodes = 1
    env._max_episode_steps = max_env_steps
    DDQN_agent = agent
    rewards = []  # Store rewards for graphing
    epsilons = []  # Store the Explore/Exploit

    # Testing agent
    for e_test in range(TEST_Episodes):
        state = env.reset()
        state = np.reshape(state, [1, s_size])
        tot_rewards = 0
        for t_test in range(max_env_steps):
            action = DDQN_agent.test_action(state)
            next_state, reward, done, _ = env.step(action)
            if done or t_test == max_env_steps - 1:
                rewards.append(tot_rewards)
                epsilons.append(0)  # We are doing full exploit
                st.write(f"episode: {e_test}/{TEST_Episodes}, score: {tot_rewards}, e: {DDQN_agent.epsilon}")
                break
            next_state = np.reshape(next_state, [1, s_size])
            tot_rewards += reward

            st.write(f"The state is: {state}, action taken is: {action}, obtained reward is: {reward}")
            # DON'T STORE ANYTHING DURING TESTING
            state = next_state

