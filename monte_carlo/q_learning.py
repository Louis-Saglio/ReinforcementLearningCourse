import random
from time import time

import gym
import numpy as np

from utils import evaluate

EPISODE_SIZE = 300
EPISODE_NUMBER = 5_000

EPSILON = 0.3
LEARNING_RATE = 0.8
DISCOUNT_FACTOR = 0.95


def train(env):
    """
    Build a Q-table which can be later used to solve the problem given by env
    :param env: The environment to solve
    :return: A Q-table trained to solve the problem given by env
    """

    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    for episode_index in range(EPISODE_NUMBER):
        previous_state = env.reset()
        for step_index in range(EPISODE_SIZE):

            if random.random() < EPSILON:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[previous_state, :])

            current_state, current_state_reward, done, info = env.step(action)

            # Uncomment the following block if you want to solve FrozenLake 8x8
            # You need to do this because in this environment,
            # the probability of getting a reward when playing at random is almost 0
            # So your Q-table is never updated, and the agent does not learn.
            # Here, if the agent is on a hole and, instead of ending the simulation,
            # the agent is given a negative reward, so it will learn not to go on holes
            if done and current_state_reward == 0:
                current_state_reward = -1
                done = False

            q_table[previous_state, action] += LEARNING_RATE * (
                current_state_reward
                + DISCOUNT_FACTOR * np.max(q_table[current_state, :])
                - q_table[previous_state, action]
            )

            previous_state = current_state

            if done:
                break

    return q_table


if __name__ == "__main__":

    def main():
        # env, winning_reward = gym.make("FrozenLake-v0", is_slippery=False), 1
        env, winning_reward = gym.make("FrozenLake8x8-v0", is_slippery=False), 1
        # env, winning_reward = gym.make("Taxi-v3"), 20

        seed = 0 or int(time())
        env.action_space.np_random.seed(seed)
        env.seed(seed)
        random.seed(seed)

        q_table = train(env)

        evaluate(
            env,
            100,
            q_table=q_table,
            winning_reward=winning_reward,
            render=True,
            display_result=True,
        )
        evaluate(
            env, 100, is_random=True, winning_reward=winning_reward, display_result=True
        )

        return env, q_table

    environment, Q = main()


# Average score with QTable in FL is either 0 or 100 because there is no random
# With TD its variable because the starting state is random
# Pourquoi ça plafonne à -20
# Pourquoi sur TD EPSILON doit être 0 avec sarsa ?
