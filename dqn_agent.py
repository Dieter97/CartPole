import random
from collections import deque

import gym

from time import sleep
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np

class DqnAgent:

    def __init__(self, env, training):
        self.env = env
        self.training = training

        self.experienceBuffer = deque(maxlen=2000)

        # constants
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # create neural network
        state_shape = self.env.observation_space.shape
        action_shape = self.env.action_space.n
        self.model = Sequential()
        self.model.add(Dense(16, input_dim=state_shape[0], activation="relu"))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dense(16, activation="relu"))
        self.model.add(Dense(action_shape))
        self.model.compile(loss="mean_squared_error",
                           optimizer=Adam(lr=self.learning_rate))

    def remember(self, state, action, reward, next_state, done):
        self.experienceBuffer.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon and self.training:
            return self.env.action_space.sample()
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        states = []
        targets = []
        if len(self.experienceBuffer) <= batch_size:
            batch_size = len(self.experienceBuffer)
        minibatch = random.sample(self.experienceBuffer, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            states.append(state[0])
            targets.append(target_f[0])

        #Train in batch to make sure the noise is minimal
        self.model.fit(np.asarray(states), np.asarray(targets), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, name):
        self.model.save(name)

    def load_model(self, name):
        self.model = load_model(name)


def run_agent(env, training=False, number_of_episodes=100, model_name=None):
    total_reward = 0
    agent = DqnAgent(env, training)

    if not training:
        try:
            if model_name is None:
                agent.load_model("{}.model".format(env.spec.id.lower()))
            else:
                agent.load_model(model_name)
        except:
            print("Failed to load {}".format(env.spec.id.lower()))
            return

    for episode in range(number_of_episodes):
        done = False
        total_episode_reward = 0
        state = env.reset()
        state = np.reshape(state, [1, 4])

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            next_state = np.reshape(next_state, [1, 4])
            agent.remember(state=state,
                           action=action,
                           reward=reward,
                           next_state=next_state,
                           done=done)

            if not training:
                env.render()
                sleep(0.02)
            state = next_state
            total_episode_reward += reward

            if training:
                agent.replay(32)
        print("Total reward for episode {} is {}".format(episode, total_episode_reward))
        total_reward += total_episode_reward

    if training:
        agent.save_model("{}.model".format(env.spec.id.lower()))
        print("Total training reward for agent after {} episodes is {}".format(number_of_episodes, total_reward))
    else:
        print("Result of {} = {}".format(env.spec.id, total_reward))
        print("Average of {} = {}".format(env.spec.id, total_reward/number_of_episodes))


def main():
    env = gym.make("CartPole-v1")

    # Train the agent
    run_agent(env, training=True, number_of_episodes=100)

    # Test performance of the agent
    run_agent(env, training=False, number_of_episodes=10)

     # Demo
    #run_agent(env, training=False, number_of_episodes=100, model_name="wiebel.model")


if __name__ == "__main__":
    main()
