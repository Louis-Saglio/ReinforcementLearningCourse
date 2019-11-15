import time

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


# Start
def main():
    env = gym.make("FrozenLake-v0")
    env.reset()
    epsilon = 0.5
    total_episodes = 100
    max_steps = 10
    lr_rate = 0.81
    gamma = 0.96
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for episode in range(total_episodes):
        print(f"\r{episode}", end="")
        state = env.reset()
        t = 0
        while t < max_steps:
            env.render()
            action = choose_action(state, epsilon, env, Q)
            state2, reward, done, info = env.step(action)
            learn(state, state2, reward, action, Q, gamma, lr_rate)
            state = state2
            t += 1
            if done:
                break
            time.sleep(0.1)

    print(Q)


if __name__ == "__main__":
    main()
