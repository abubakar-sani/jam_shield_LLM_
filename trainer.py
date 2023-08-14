#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import json
import streamlit as st
from DDQN import DoubleDeepQNetwork
from antiJamEnv import AntiJamEnv


def train(jammer_type, channel_switching_cost):
    env = AntiJamEnv(jammer_type, channel_switching_cost)
    ob_space = env.observation_space
    ac_space = env.action_space
    st.write(f"Observation space: , {ob_space}")
    st.write(f"Action space: {ac_space}")

    s_size = ob_space.shape[0]
    a_size = ac_space.n
    max_env_steps = 100
    TRAIN_Episodes = 10
    env._max_episode_steps = max_env_steps

    epsilon = 1.0  # exploration rate
    epsilon_min = 0.01
    epsilon_decay = 0.999
    discount_rate = 0.95
    lr = 0.001
    batch_size = 32

    DDQN_agent = DoubleDeepQNetwork(s_size, a_size, lr, discount_rate, epsilon, epsilon_min, epsilon_decay)
    rewards = []  # Store rewards for graphing
    epsilons = []  # Store the Explore/Exploit

    # Training agent
    for e in range(TRAIN_Episodes):
        state = env.reset()
        # print(f"Initial state is: {state}")
        state = np.reshape(state, [1, s_size])  # Resize to store in memory to pass to .predict
        tot_rewards = 0
        for time in range(max_env_steps):  # 200 is when you "solve" the game. This can continue forever as far as I know
            action = DDQN_agent.action(state)
            next_state, reward, done, _ = env.step(action)
            # print(f'The next state is: {next_state}')
            # done: Three collisions occurred in the last 10 steps.
            # time == max_env_steps - 1 : No collisions occurred
            if done or time == max_env_steps - 1:
                rewards.append(tot_rewards)
                epsilons.append(DDQN_agent.epsilon)
                st.write(f"episode: {e}/{TRAIN_Episodes}, score: {tot_rewards}, e: {DDQN_agent.epsilon}")
                break
            # Applying channel switching cost
            next_state = np.reshape(next_state, [1, s_size])
            tot_rewards += reward
            DDQN_agent.store(state, action, reward, next_state, done)  # Resize to store in memory to pass to .predict
            state = next_state

            # Experience Replay
            if len(DDQN_agent.memory) > batch_size:
                DDQN_agent.experience_replay(batch_size)
        # Update the weights after each episode (You can configure this for x steps as well
        DDQN_agent.update_target_from_model()
        # If our current NN passes we are done
        # Early stopping criteria: I am going to use the last 10 runs within 1% of the max
        if len(rewards) > 10 and np.average(rewards[-10:]) >= max_env_steps - 0.10 * max_env_steps:
            break

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
    plt.title(f'Training Rewards - {jammer_type}, CSC: {channel_switching_cost}')
    plt.legend()

    # Display the Streamlit figure using streamlit.pyplot
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig)

    # Save the figure
    plot_name = f'train_rewards_{jammer_type}_csc_{channel_switching_cost}.png'
    plt.savefig(plot_name, bbox_inches='tight')
    plt.close(fig)  # Close the figure to release resources

    # Save Results
    # Rewards
    fileName = f'train_rewards_{jammer_type}_csc_{channel_switching_cost}.json'
    with open(fileName, 'w') as f:
        json.dump(rewards, f)

    # Save the agent as a SavedAgent.
    agentName = f'DDQNAgent_{jammer_type}_csc_{channel_switching_cost}'
    DDQN_agent.save_model(agentName)
