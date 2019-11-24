import random
from time import time

import gym
import numpy as np

EPISODE_SIZE = 300
EPISODE_NUMBER = 7_000

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
            # if done and current_state_reward == 0:
            #     current_state_reward = -1
            #     done = False

            q_table[previous_state, action] += LEARNING_RATE * (
                current_state_reward
                + DISCOUNT_FACTOR * np.max(q_table[current_state, :])
                - q_table[previous_state, action]
            )

            previous_state = current_state

            if done:
                break

    return q_table


def evaluate(
    env,
    total_episodes,
    *,
    q_table=None,
    winning_reward=None,
    is_random=False,
    render=False,
):
    """
    Evaluate the performance of a q-table to solve a gym environment problem
    It may also use random instead of a q-table
    in order to compare the performance of a q-table against a random solution
    :param env: gym environment to solve
    :param total_episodes: number of time to repeat the evaluation.
           The bigger the more statistically significant the output will be
    :param q_table: Q-table to used solve the problem
           if given, is_random must be False
    :param winning_reward: the reward given to the agent when it solves the problem.
           It is used to compute the number of time the agent solved the problem
    :param is_random: if True will use random instead of Q-table.
           If True, q-table must not be given
    :param render: if True will call env.render()
    """

    if (q_table is not None) and is_random:
        raise RuntimeError("is_random and q_table given")
    elif q_table is None and is_random is None:
        raise RuntimeError("at least one of q_table and is_random must be given")

    total_epochs, total_reward, total_won_episodes = 0, 0, 0

    for _ in range(total_episodes):
        state = env.reset()
        if render:
            env.render()
        done = False
        while not done:
            if is_random:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])
            state, reward, done, info = env.step(action)

            total_epochs += 1
            total_reward += reward

            if render:
                env.render()

        # noinspection PyUnboundLocalVariable
        if reward == winning_reward:
            total_won_episodes += 1

    print("-" * 30)
    print(
        f"Results after {total_episodes} episodes using {'random' if is_random else 'q_table'}:"
    )
    print(f"Average steps per episode: {total_epochs / total_episodes}")
    print(f"Average reward per episode: {total_reward / total_episodes}")
    print(
        f"Percentage of won episodes : {round(total_won_episodes * 100 / total_episodes, 2)}%"
    )


if __name__ == "__main__":

    def main():
        env, winning_reward = gym.make("FrozenLake-v0", is_slippery=False), 1
        # env, winning_reward = gym.make("FrozenLake8x8-v0", is_slippery=False), 1
        # env, winning_reward = gym.make("Taxi-v3"), 20

        seed = 0 or int(time())
        env.action_space.np_random.seed(seed)
        env.seed(seed)
        random.seed(seed)

        q_table = train(env)

        evaluate(env, 100, q_table=q_table, winning_reward=winning_reward, render=True)
        evaluate(env, 100, is_random=True, winning_reward=winning_reward)

        return env, q_table

    environment, Q = main()
