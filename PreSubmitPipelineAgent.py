import collections
import random

import numpy as np
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from Evaluator import Evaluator
from ci_simulation.CISimulatorPre import CISimulatorPre
from ci_simulation.DatabasePre import DatabasePre
from helpers import seed


def DNN(input_size):
    """
    Deep Neural Network for predicting the Q value of the action 'execute test case'.

    Parameters:
    input_size (tuple): Input shape for the neural network.

    Returns:
    keras.Model: Compiled Keras model.
    """
    model = Sequential()
    model.add(Dense(1024, input_shape=input_size, activity_regularizer=regularizers.L2(0.125)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(256, activity_regularizer=regularizers.L2(0.125)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activity_regularizer=regularizers.L2(0.125)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.35))
    model.add(Dense(1, activation='tanh', activity_regularizer=regularizers.L2(0.125)))

    # Optimizer
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=2e-4,
        decay_steps=100000,
        decay_rate=0.95, 
        staircase=False
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=0.3, clipnorm=1.0)

    # Compile model
    model.compile(
        loss="mean_squared_error",
        optimizer=optimizer,
    )

    return model

class DQNAgent:
    def __init__(self, database, evaluator, resource_budget, epsilon, pca_components=9):
        """
        DQN Agent for reinforcement learning in CI simulation environment.

        Parameters:
        database (DatabasePre): The database mock object.
        evaluator (Evaluator): The evaluator object.
        resource_budget (float): Available resource budget.
        epsilon (tuple): Initial and minimum epsilon values for exploration.
        """
        self.evaluator = evaluator
        self.env = CISimulatorPre(database, evaluator, pipeline='PRE', pca_components=pca_components)
        self.env.available_resources = resource_budget

        self.state_size = len(self.env.features)
        self.episodes = len(database.job_data)

        # Set hyperparameters
        self.memory = collections.deque(maxlen=4096)
        self.epsilon = epsilon[0]
        self.epsilon_min = epsilon[1]
        self.epsilon_decay = 0.999
        self.batch_size = 128
        self.train_start = 2048
        self.skip_jobs_without_changes = True
        self.gamma = 0.8  # discount rate

        # Model
        self.model = DNN(input_size=(self.state_size,))
        self.model.summary()

    def greedy_exploration(self, state):
        """
        Greedy Exploration: Predict reward of taking certain actions, with added randomness for exploration.

        Parameters:
        state (np.ndarray): Current state input for the model.

        Returns:
        list: List of actions with added exploration noise.
        """
        predictions = list(self.model.predict(state).flatten())

        for idx in range(len(predictions)):
            predictions[idx] += np.random.normal(0, self.epsilon)

        return predictions

    def fill_memory(self, state, action, reward, next_state, done):
        """
        Fill the replay memory with experiences.

        Parameters:
        state (np.ndarray): Current state.
        action (float): Action taken.
        reward (float): Reward received.
        next_state (np.ndarray): Next state after the action.
        done (bool): Whether the episode is finished.
        """
        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) >= self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def train(self):
        """
        Train the model with a random batch from the replay memory.

        Returns:
        keras.callbacks.History: Training history.
        """
        if len(self.memory) < self.train_start:
            return None

        memory_batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        for ind in range(self.batch_size):
            state[ind] = memory_batch[ind][0]
            action.append(memory_batch[ind][1])
            reward.append(memory_batch[ind][2])
            next_state[ind] = memory_batch[ind][3]
            done.append(memory_batch[ind][4])

        target_next = self.model.predict(next_state)

        for i in range(self.batch_size):
            if not done[i]:
                reward[i] = float(reward[i]) + float(self.gamma * target_next[i][0])

        history = self.model.fit(state, np.array(reward).astype(float), batch_size=self.batch_size, verbose=1)
        return history

    def run(self):
        """
        Run the DQN agent for a number of episodes, interacting with the environment and training the model.
        """
        for e in range(1, self.episodes):
            states = self.env.reset(e).to_numpy().astype('float32')
            if len(states) == 0:
                continue

            if self.skip_jobs_without_changes and len(self.env.observation[self.env.observation.hypothetical_reward > 0]) == 0:
                continue

            actions = self.greedy_exploration(states)
            result, done = self.env.step(actions)
            scheduled_idxs = list(result[result.scheduled].index)

            print(f"{e}: {len(result[result.test_result == 'FAILED'])}/{len(result)}")

            next_states = np.roll(states, 1, axis=0)
            for idx in scheduled_idxs:
                done = idx == scheduled_idxs[-1]
                if result.loc[idx, 'reward'] <= 0:
                    self.fill_memory(states[idx], result.loc[idx, 'prediction'], result.loc[idx, 'reward'],
                                     next_states[idx], done)

            if e < 40 or e % 128 == 0:
                self.train()

        self.evaluator.save(self.evaluator.name)

if __name__ == "__main__":
    evaluator = Evaluator('pre_submit_results')
    database = DatabasePre('dataset/preprocessed_dataset_pre_submit_public.pkl', evaluator, history=25)
    agent = DQNAgent(database, evaluator, resource_budget=0.8, epsilon=(0.8, 0.04))
    # agent = DQNAgent(database, evaluator, resource_budget=0.5, epsilon=(0.025, 0.005), pca_components=7)


    agent.run()
