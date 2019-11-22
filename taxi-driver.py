from random import random

import gym
import numpy as np

EPISODE_SIZE = 10
EPISODE_NUMBER = 100_000

EPSILON = 0.3
LEARNING_RATE = 0.81
GAMMA = 0.96


def train(env):

    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    for episode_index in range(EPISODE_NUMBER):
        state = env.reset()
        for step_index in range(EPISODE_SIZE):

            if random() < EPSILON:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            next_state, reward, done, info = env.step(action)

            q_table[state, action] = q_table[state, action] + LEARNING_RATE * (
                reward + GAMMA * np.max(q_table[next_state, :]) - q_table[state, action]
            )

            state = next_state
            # env.render()
            if done:
                break
    return q_table


def evaluate(env, total_episodes, *, q_table=None, is_random=False, render=False):
    if (q_table is not None) and is_random:
        raise RuntimeError("is_random and q_table given")
    elif q_table is None and is_random is None:
        raise RuntimeError("at least one of q_table and is_random must be given")

    total_epochs, total_rewards = 0, 0

    for _ in range(total_episodes):
        state = env.reset()
        if render:
            env.render()
        epochs, reward = 0, 0
        done = False
        while not done:
            if is_random:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])
            state, reward, done, info = env.step(action)

            epochs += 1

            if render:
                env.render()

        total_rewards += reward
        total_epochs += epochs

    print(
        f"Results after {total_episodes} episodes using {'random' if is_random else 'q_table'}:"
    )
    print(f"Average timesteps per episode: {total_epochs / total_episodes}")
    print(f"Average rewards per episode: {total_rewards / total_episodes}")


if __name__ == "__main__":

    def main():
        env = gym.make("Taxi-v3")
        q_table = train(env)
        evaluate(env, 1, q_table=q_table, render=True)
        evaluate(env, 10, is_random=True)

    main()
