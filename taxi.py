import pickle

import gym
import numpy as np


def choose_action(state, epsilon, env, Q):
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


def learn(state, state2, reward, action, Q, gamma, lr_rate):
    predict = Q[state, action]
    target = reward + gamma * np.max(Q[state2, :])
    Q[state, action] = Q[state, action] + lr_rate * (target - predict)


def evaluate(is_random, env, total_episodes, Q):
    total_epochs, total_rewards = 0, 0

    for _ in range(total_episodes):
        state = env.reset()
        epochs, reward = 0, 0
        done = False
        while not done:
            if is_random:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])
            state, reward, done, info = env.step(action)

            epochs += 1

        total_rewards += reward
        total_epochs += epochs

    print(f"Results after {total_episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / total_episodes}")
    print(f"Average rewards per episode: {total_rewards / total_episodes}")


# Start
def main(load_from=None, save_to=None):
    env = gym.make("Taxi-v3")
    env.reset()
    epsilon = 0.5
    total_episodes = 10000
    max_steps = 10
    lr_rate = 0.81
    gamma = 0.96
    if load_from is not None:
        lr_rate = 0
        with open(load_from, "rb") as f:
            Q = pickle.load(f)
    else:
        Q = np.zeros((env.observation_space.n, env.action_space.n))

    for episode in range(total_episodes):
        state = env.reset()
        t = 0
        while t < max_steps:
            action = choose_action(state, epsilon, env, Q)
            state2, reward, done, info = env.step(action)
            learn(state, state2, reward, action, Q, gamma, lr_rate)
            state = state2
            t += 1
            if done:
                break

    # print(Q)
    if save_to:
        with open(save_to, "wb") as f:
            pickle.dump(Q, f)

    evaluate(False, env, total_episodes, Q)
    evaluate(False, env, total_episodes, Q)
    evaluate(True, env, total_episodes, Q)


if __name__ == "__main__":
    main()
    # main("file", "file")
