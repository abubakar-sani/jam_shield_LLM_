import streamlit as st

def main():
    st.title("Anti-Jamming Configuration App")
    mode = st.radio("Choose Mode", ["Auto", "Manual"])

    if mode == "Auto":
        jammer_type = "dynamic"
        agent_type = "DQN with prioritized replay memory"
        channel_switching_cost = 0.1
    else:
        jammer_type = st.selectbox("Select Jammer Type", ["constant", "sweeping", "random", "dynamic"])
        agent_type = st.selectbox("Select Agent Type", ["DQN", "DQN with fixed targets", "DDQN", "Dueling DDQN", "DQN with prioritized replay memory"])
        channel_switching_cost = st.selectbox("Select Channel Switching Cost", [0, 0.05, 0.1, 0.15, 0.2])

    st.write("Configuration:")
    st.write(f"Jammer Type: {jammer_type}")
    st.write(f"Agent Type: {agent_type}")
    st.write(f"Channel Switching Cost: {channel_switching_cost}")

if __name__ == "__main__":
    main()
