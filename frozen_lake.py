import gym
import random
import numpy as np
import time, pickle, os

ENV = gym.make("FrozenLake-v0")
ENV.reset()
TOTAL_EPISODES = 10_000
MAS_STEPS = 100
epsilon = 0.9
lr_rate = 0.81
gamma = 0.96

q = np.zeros((ENV.observation_space.n, ENV.action_space.n))
for n in range(TOTAL_EPISODES):
    lr_rate, epsilon = 1, 1
    current_state = ENV.reset()
    for i in range(MAS_STEPS):
        s = current_state
        nb = random.randint(0, 1)
        if nb < epsilon:
            a = ENV.action_space.sample()
        else:
            a = np.argmax(q)
        next_state, reward, done, _ = ENV.step(a)
        q = lr_rate * (reward * gamma + np.argmax(q))
        lr_rate *= 0.99
        epsilon *= 0.99

        if done:
            break
