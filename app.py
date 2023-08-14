#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import os
from trainer import train
from tester import test


def main():
    st.title("Beyond the Anti-Jam: Integration of DRL with LLM")

    st.sidebar.header("Make Your Environment Configuration")
    mode = st.sidebar.radio("Choose Mode", ["Auto", "Manual"])

    if mode == "Auto":
        jammer_type = "dynamic"
        channel_switching_cost = 0.1
    else:
        jammer_type = st.sidebar.selectbox("Select Jammer Type", ["constant", "sweeping", "random", "dynamic"])
        channel_switching_cost = st.sidebar.selectbox("Select Channel Switching Cost", [0, 0.05, 0.1, 0.15, 0.2])

    st.sidebar.subheader("Configuration:")
    st.sidebar.write(f"Jammer Type: {jammer_type}")
    st.sidebar.write(f"Channel Switching Cost: {channel_switching_cost}")

    train_button = st.sidebar.button('Train')
    test_button = st.sidebar.button('Test')

    if train_button or test_button:
        agent_name = f'DDQNAgent_{jammer_type}_csc_{channel_switching_cost}'
        if os.path.exists(agent_name):
            if train_button:
                st.warning("Agent has been trained already! Do you want to retrain?")
                retrain = st.sidebar.button('Yes')
                if retrain:
                    perform_training(jammer_type, channel_switching_cost)
            elif test_button:
                perform_testing(jammer_type, channel_switching_cost)
        else:
            if train_button:
                perform_training(jammer_type, channel_switching_cost)
            elif test_button:
                st.warning("Agent has not been trained yet. Click Train First!!!")


def perform_training(jammer_type, channel_switching_cost):
    st.sidebar.write("==================================================")
    st.sidebar.write('Training Starting')
    train(jammer_type, channel_switching_cost)
    st.sidebar.write("Training completed")
    st.sidebar.write("==================================================")


def perform_testing(jammer_type, channel_switching_cost):
    st.sidebar.write("==================================================")
    st.sidebar.write('Testing Starting')
    test(jammer_type, channel_switching_cost)
    st.sidebar.write("Testing completed")
    st.sidebar.write("==================================================")


if __name__ == "__main__":
    main()
