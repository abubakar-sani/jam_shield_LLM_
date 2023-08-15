#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import json
import streamlit as st
from DDQN import DoubleDeepQNetwork
from antiJamEnv import AntiJamEnv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "tiiuae/falcon-7b-instruct"  # Replace with the exact model name or path
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def train(jammer_type, channel_switching_cost):
    st.subheader("DRL Training Progress")
    progress_bar = st.progress(0)
    status_text = st.empty()

    env = AntiJamEnv(jammer_type, channel_switching_cost)
    ob_space = env.observation_space
    ac_space = env.action_space

    s_size = ob_space.shape[0]
    a_size = ac_space.n
    max_env_steps = 100
    TRAIN_Episodes = 5
    env._max_episode_steps = max_env_steps

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.999
    discount_rate = 0.95
    lr = 0.001
    batch_size = 32

    DDQN_agent = DoubleDeepQNetwork(s_size, a_size, lr, discount_rate, epsilon, epsilon_min, epsilon_decay)
    rewards = []
    epsilons = []

    for e in range(TRAIN_Episodes):
        state = env.reset()
        state = np.reshape(state, [1, s_size])
        tot_rewards = 0
        for time in range(max_env_steps):
            action = DDQN_agent.action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, s_size])
            tot_rewards += reward
            DDQN_agent.store(state, action, reward, next_state, done)
            state = next_state

            if len(DDQN_agent.memory) > batch_size:
                DDQN_agent.experience_replay(batch_size)

            if done or time == max_env_steps - 1:
                rewards.append(tot_rewards)
                epsilons.append(DDQN_agent.epsilon)
                status_text.text(
                    f"Episode: {e + 1}/{TRAIN_Episodes}, Reward: {tot_rewards}, Epsilon: {DDQN_agent.epsilon:.3f}")
                progress_bar.progress((e + 1) / TRAIN_Episodes)
                break

        DDQN_agent.update_target_from_model()

        if len(rewards) > 10 and np.average(rewards[-10:]) >= max_env_steps - 0.10 * max_env_steps:
            break

    st.sidebar.success("DRL Training completed!")

    # Plotting
    rolling_average = np.convolve(rewards, np.ones(10) / 10, mode='valid')
    solved_threshold = max_env_steps - 0.10 * max_env_steps
    # Create a new Streamlit figure for the training graph
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rewards, label='Rewards')
    ax.plot(rolling_average, color='black', label='Rolling Average')
    ax.axhline(y=solved_threshold, color='r', linestyle='-', label='Solved Line')
    eps_graph = [100 * x for x in epsilons]
    ax.plot(eps_graph, color='g', linestyle='-', label='Epsilons')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Rewards')
    ax.set_title(f'Training Rewards - {jammer_type}, CSC: {channel_switching_cost}')
    ax.legend()

    insights = generate_insights(rewards, rolling_average, epsilons, solved_threshold)

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Training Graph")
            st.pyplot(fig)

        with col2:
            st.subheader("Graph Explanation")
            st.write(insights)

    # Save the figure
    # plot_name = f'./data/train_rewards_{jammer_type}_csc_{channel_switching_cost}.png'
    # plt.savefig(plot_name, bbox_inches='tight')
    plt.close(fig)  # Close the figure to release resources

    # Save Results
    # Rewards
    # fileName = f'./data/train_rewards_{jammer_type}_csc_{channel_switching_cost}.json'
    # with open(fileName, 'w') as f:
    #     json.dump(rewards, f)
    #
    # # Save the agent as a SavedAgent.
    # agentName = f'./data/DDQNAgent_{jammer_type}_csc_{channel_switching_cost}'
    # DDQN_agent.save_model(agentName)
    return DDQN_agent


def generate_insights(rewards, rolling_average, epsilons, solved_threshold):
    description = (
        f"The graph represents training rewards over episodes. "
        f"The actual rewards range from {min(rewards)} to {max(rewards)} with an average of {np.mean(rewards):.2f}. "
        f"The rolling average values range from {min(rolling_average)} to {max(rolling_average)} with an average of {np.mean(rolling_average):.2f}. "
        f"The epsilon values range from {min(epsilons)} to {max(epsilons)} with an average exploration rate of {np.mean(epsilons):.2f}. "
        f"The solved threshold is set at {solved_threshold}. "
        f"Provide insights based on this data."
    )
    input_ids = tokenizer.encode(description, return_tensors="pt")

    # Generate output from model
    output_ids = model.generate(input_ids, max_length=300, num_return_sequences=1)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output_text
