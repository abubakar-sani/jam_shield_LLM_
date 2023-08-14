#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import json
import streamlit as st
from DDQN import DoubleDeepQNetwork
from antiJamEnv import AntiJamEnv


def test(jammer_type, channel_switching_cost):
    env = AntiJamEnv(jammer_type, channel_switching_cost)
    ob_space = env.observation_space
    ac_space = env.action_space

    s_size = ob_space.shape[0]
    a_size = ac_space.n
    max_env_steps = 100
    TEST_Episodes = 10
    env._max_episode_steps = max_env_steps

    epsilon = 1.0  # exploration rate
    epsilon_min = 0.01
    epsilon_decay = 0.999
    discount_rate = 0.95
    lr = 0.001

    agentName = f'./data/DDQNAgent_{jammer_type}_csc_{channel_switching_cost}'
    DDQN_agent = DoubleDeepQNetwork(s_size, a_size, lr, discount_rate, epsilon, epsilon_min, epsilon_decay)
    DDQN_agent.model = DDQN_agent.load_saved_model(agentName)
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
            # DON'T STORE ANYTHING DURING TESTING
            state = next_state

    # Plotting
    rolling_average = np.convolve(rewards, np.ones(10) / 10, mode='valid')

    # Create a new Streamlit figure
    fig = plt.figure()
    plt.plot(rewards, label='Rewards')
    plt.plot(rolling_average, color='black', label='Rolling Average')
    plt.axhline(y=max_env_steps - 0.10 * max_env_steps, color='r', linestyle='-', label='Solved Line')
    eps_graph = [100 * x for x in epsilons]
    plt.plot(eps_graph, color='g', linestyle='-', label='Epsilons')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title(f'Testing Rewards - {jammer_type}, CSC: {channel_switching_cost}')
    plt.legend()

    # Display the Streamlit figure using streamlit.pyplot
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig)

    # Save the figure
    plot_name = f'./data/test_rewards_{jammer_type}_csc_{channel_switching_cost}.png'
    plt.savefig(plot_name, bbox_inches='tight')
    plt.close(fig)  # Close the figure to release resources

    # Save Results
    # Rewards
    fileName = f'./data/test_rewards_{jammer_type}_csc_{channel_switching_cost}.json'
    with open(fileName, 'w') as f:
        json.dump(rewards, f)
