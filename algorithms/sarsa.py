import random
from time import time

import gym
import numpy as np

from q_learning import evaluate

EPISODE_SIZE = 300
EPISODE_NUMBER = 100

EPSILON = 0.3
LEARNING_RATE = 0.8
DISCOUNT_FACTOR = 0.95


def choose_action(env, q_table, from_state):
    if random.random() < EPSILON:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[from_state, :])
    return action


def train(env, q_table=None):
    """
    Build a Q-table, which can be later used to solve the problem given by env
    :param env: The environment to solve
    :param q_table: Start with this q-table. If None, will build a zero filled q-table
    :return: A Q-table trained to solve the problem given by env
    """

    if q_table is None:
        q_table = np.zeros((env.observation_space.n, env.action_space.n))

    while True:
        for episode_index in range(EPISODE_NUMBER):
            previous_state = env.reset()
            previous_action = choose_action(env, q_table, previous_state)
            for step_index in range(EPISODE_SIZE):

                current_state, current_state_reward, done, info = env.step(
                    previous_action
                )

                next_action = choose_action(env, q_table, current_state)

                q_table[previous_state, previous_action] += LEARNING_RATE * (
                    current_state_reward
                    + DISCOUNT_FACTOR * q_table[current_state, next_action]
                    - q_table[previous_state, previous_action]
                )

                previous_state = current_state
                previous_action = next_action

                if done:
                    break

        yield q_table


def main():
    env, winning_reward = gym.make("FrozenLake-v0", is_slippery=False), 1
    # env, winning_reward = gym.make("FrozenLake8x8-v0", is_slippery=False), 1
    # env, winning_reward = gym.make("Taxi-v3"), 20

    seed = 0 or int(time())
    env.action_space.np_random.seed(seed)
    env.seed(seed)
    random.seed(seed)

    q_table_generator = train(env)
    q_table = next(q_table_generator)
    while (
        evaluate(
            env,
            100,
            q_table=q_table,
            winning_reward=winning_reward,
            render=False,
            display_result=True,
        )
        < 90
    ):
        q_table = next(q_table_generator)

    evaluate(
        env, 100, is_random=True, winning_reward=winning_reward, display_result=True
    )

    return env, q_table


if __name__ == "__main__":
    environment, Q = main()
