import pickle

import gym
import numpy as np


def choose_action(state, epsilon, env, Q):
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


def learn_sarsa(state, state2, reward, action, action2, Q, gamma, lr_rate):
    predicted_action = Q[state, action]
    Q[state, action] = predicted_action + lr_rate * (
        reward + gamma * Q[state2, action2] - predicted_action
    )


def learn_qlearning(state, state2, reward, action, Q, gamma, lr_rate):
    predict = Q[state, action]
    Q[state, action] = predict + lr_rate * (
        reward + gamma * np.max(Q[state2, :]) - predict
    )


# Start
def main(load_from=None, save_to=None):
    if load_from is not None:
        lr_rate = 0
        with open(load_from, "rb") as f:
            Q = pickle.load(f)
    else:
        Q = np.zeros((env.observation_space.n, env.action_space.n))

    for episode in range(total_episodes):
        print(f"\r{episode}", end="")
        state = env.reset()
        t = 0
        while t < max_steps:
            # env.render()
            action = choose_action(state, epsilon, env, Q)
            state2, reward, done, info = env.step(action)
            action2 = choose_action(state2, 0, env, Q)
            learn_sarsa(state, state2, reward, action, action2, Q, gamma, lr_rate)
            # learn_qlearning(state, state2, reward, action, Q, gamma, lr_rate)
            state = state2
            t += 1
            if done:
                break

    print(Q)
    if save_to:
        with open(save_to, "wb") as f:
            pickle.dump(Q, f)

    return Q


def evaluate(Q):
    total_epochs, total_penalties = 0, 0

    for _ in range(total_episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0

        done = False

        while not done:
            action = np.argmax(Q[state, :])
            state, reward, done, info = env.step(action)

            if reward < 0:
                penalties += 1

            epochs += 1

        total_penalties += penalties
        total_epochs += epochs

    print(f"Results after {total_episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / total_episodes}")
    print(f"Average penalties per episode: {total_penalties / total_episodes}")


if __name__ == "__main__":
    # main()
    env = gym.make("FrozenLake-v0")
    env.reset()
    epsilon = 0.5
    total_episodes = 10000
    max_steps = 10
    lr_rate = 0.81
    gamma = 0.96
    q = main("file", "file")
    evaluate(q)
