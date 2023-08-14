import streamlit as st

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

    st.write("==================================================")
    st.write("Training Starting")
    st.write("Training completed")
    st.write("==================================================")
    st.write("")

if __name__ == "__main__":
    main()
