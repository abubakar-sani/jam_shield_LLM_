#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import os
from trainer import train
from tester import test
import transformers
from transformers import TFAutoModelForCausalLM, AutoTokenizer


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

    start_button = st.sidebar.button('Start')

    if start_button:
        agent = perform_training(jammer_type, channel_switching_cost)
        test(agent, jammer_type, channel_switching_cost)

def perform_training(jammer_type, channel_switching_cost):
    agent = train(jammer_type, channel_switching_cost)
    st.subheader("Generating Insights of the DRL-Training")
    model_name = "tiiuae/falcon-7b-instruct"
    model = TFAutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = TFAutoTokenizer.from_pretrained(model_name)
    pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=100, temperature=0.7)
    text = pipeline("Discuss this topic: Integrating LLMs to DRL-based anti-jamming.")
    st.write(text)
    return agent


def perform_testing(agent, jammer_type, channel_switching_cost):
    test(agent, jammer_type, channel_switching_cost)


if __name__ == "__main__":
    main()
