#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from collections import deque
from tensorflow import keras
from tensorflow.keras.models import load_model
import random
import streamlit


class DoubleDeepQNetwork:
    def __init__(self, s_size, a_size, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        self.nS = s_size
        self.nA = a_size
        self.memory = deque([], maxlen=2500)
        self.alpha = alpha
        self.gamma = gamma
        # Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()
        self.model_target = self.build_model()  # Second (target) neural network
        self.update_target_from_model()  # Update weights
        self.loss = []

    def build_model(self):
        model = keras.Sequential()  # linear stack of layers https://keras.io/models/sequential/
        model.add(keras.layers.Dense(256, input_dim=self.nS, activation='relu'))  # [Input] -> Layer 1
        model.add(keras.layers.Dense(256, activation='relu'))  # Layer 2 -> 3
        model.add(keras.layers.Dense(self.nA, activation='linear'))  # Layer 3 -> [output]

        model.compile(loss='mean_squared_error',  # Loss function: Mean Squared Error
                      optimizer=keras.optimizers.Adam(
                          lr=self.alpha))  # Optimizer: Adam (Feel free to check other options)
        return model

    def update_target_from_model(self):
        # Update the target model from the base model
        self.model_target.set_weights(self.model.get_weights())

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nA)  # Explore
        action_vals = self.model.predict(state)  # Exploit: Use the NN to predict the correct action from this state
        return np.argmax(action_vals[0])

    def test_action(self, state):  # Exploit
        action_vals = self.model.predict(state)
        return np.argmax(action_vals[0])

    def store(self, state, action, reward, next_state, done):
        # Store the experience in memory
        self.memory.append((state, action, reward, next_state, done))

    def save_model(self, agentName):
        # Save the agent model weights in a file
        self.model.save(agentName)

    def load_saved_model(self, agent_name):
        return load_model(agent_name)

    def experience_replay(self, batch_size):
        # Execute the experience replay
        minibatch = random.sample(self.memory, batch_size)  # Randomly sample from memory
        # streamlit.write(f"{minibatch}")
        # Convert to numpy for speed by vectorization
        x = []
        y = []
        np_array = list(minibatch)
        st = np.zeros((0, self.nS))  # States
        nst = np.zeros((0, self.nS))  # Next States
        for i in range(len(np_array)):
            st = np.append(st, np_array[i][0], axis=0)
            nst = np.append(nst, np_array[i][3], axis=0)

        st_predict = self.model.predict(st)  # Here is the speedup! I can predict on the ENTIRE batch
        nst_predict = self.model.predict(nst)
        nst_predict_target = self.model_target.predict(nst)  # Predict from the TARGET
        index = 0
        for state, action, reward, next_state, done in minibatch:
            x.append(state)
            # Predict from state
            nst_action_predict_target = nst_predict_target[index]
            nst_action_predict_model = nst_predict[index]
            if done:  # Terminal: Just assign reward much like {* (not done) - QB[state][action]}
                target = reward
            else:  # Non-terminal
                target = reward + self.gamma * nst_action_predict_target[
                    np.argmax(nst_action_predict_model)]  # Using Q to get T is Double DQN
            target_f = st_predict[index]
            target_f[action] = target
            y.append(target_f)
            index += 1
        # Reshape for Keras Fit
        x_reshape = np.array(x).reshape(batch_size, self.nS)
        y_reshape = np.array(y)
        epoch_count = 1
        hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)
        # Graph Losses
        for i in range(epoch_count):
            self.loss.append(hist.history['loss'][i])
        # Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
