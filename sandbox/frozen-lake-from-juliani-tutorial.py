"""
Source : https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
"""
from random import random, seed

import gym
import numpy as np
from taxi_driver import evaluate

env = gym.make("FrozenLake-v0")

env.action_space.np_random.seed(42)
env.seed(42)
seed(42)

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Set learning parameters
lr = 0.8
y = 0.95
num_episodes = 2000
# create lists to contain total rewards and steps per episode
# jList = []
rList = []

for i in range(num_episodes):
    # Reset environment and get first new observation
    # noinspection PyRedeclaration
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    # The Q-Table learning algorithm
    while j < 99:
        j += 1

        # Choose an action by greedily (with noise) picking from Q table
        # a = np.argmax(
        #     Q[s, :] + np.random.randn(1, env.action_space.n) * (1.0 / (i + 1))
        # )

        if random() < 0.3:
            a = env.action_space.sample()
        else:
            a = np.argmax(Q[s, :])

        # Get new state and reward from environment
        s1, r, d, _ = env.step(a)
        # Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1, :]) - Q[s, a])
        rAll += r
        s = s1
        if d == True:
            break
    # jList.append(j)
    rList.append(rAll)


if __name__ == "__main__":
    evaluate(env, 100, q_table=Q, winning_reward=1)
    evaluate(env, 100, is_random=True, winning_reward=1)
