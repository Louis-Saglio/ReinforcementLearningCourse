import gym
import numpy as np

epsilon = 0.9
nb_episode = 10000
nb_step = 100
learning_rate = 0.81
gamma = 0.96


class FrozenLake:
    def __init__(self):
        self.env = gym.make("FrozenLake8x8-v0")
        self.env.reset()
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def choice_action(self, state):
        return (
            self.env.action_space.sample()
            if np.random.uniform(0, 1) < epsilon
            else np.argmax(self.Q[state, :])
        )

    def learn(self, state, state2, reward, action):
        predict = self.Q[state, action]
        target = reward + gamma * np.max(self.Q[state2, :])
        self.Q[state, action] = self.Q[state, action] + learning_rate * (
            target - predict
        )

    def run(self):
        for _ in range(nb_episode):
            state = self.env.reset()

            for __ in range(nb_step):
                # self.env.render()
                action = self.choice_action(state)
                state2, reward, done, info = self.env.step(action)
                self.learn(state, state2, reward, action)

                state = state2

                if done:
                    break

        print(self.Q)


if __name__ == "__main__":
    frozen_lake = FrozenLake()
    frozen_lake.run()
