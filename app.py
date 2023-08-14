#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import streamlit as st
import trainer
import tester
import os


def main():
    st.title("Beyond the Anti-Jam: Integration of DRL with LLM")
    st.header("Make Your Environment Configuration")
    mode = st.radio("Choose Mode", ["Auto", "Manual"])

    if mode == "Auto":
        jammer_type = "dynamic"
        channel_switching_cost = 0.1
    else:
        jammer_type = st.selectbox("Select Jammer Type", ["constant", "sweeping", "random", "dynamic"])
        channel_switching_cost = st.selectbox("Select Channel Switching Cost", [0, 0.05, 0.1, 0.15, 0.2])

    st.subheader("Configuration:")
    st.write(f"Jammer Type: {jammer_type}")
    st.write(f"Channel Switching Cost: {channel_switching_cost}")

    if st.button('Train'):
        st.write("==================================================")
        st.write('Training Starting')
        trainer.train(jammer_type, channel_switching_cost)
        st.write("Training completed")
        st.write("==================================================")

    if st.button('Test'):
        st.write("==================================================")
        st.write('Testing Starting')
        agentName = f'savedAgents/DDQNAgent_{jammer_type}_csc_{channel_switching_cost}'
        if os.path.exists(agentName):
            tester.test(jammer_type, channel_switching_cost)
            st.write("Testing completed")
            st.write("==================================================")
        else:
            st.write("Agent has not been trained yet. Click Train First!!!")


if __name__ == "__main__":
    main()
