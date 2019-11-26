import random
from time import time
from typing import Dict, Any

import gym
import numpy as np
from gym.wrappers import TimeLimit


class Problem:
    def __init__(
        self,
        env_name: str,
        env_kwargs: Dict[str, Any],
        epsilon: float,
        learning_rate: float,
        discount_factor: float,
        episode_before_evaluation: int,
        episode_maximum_size: int,
        minimum_accepted_score: float,
        winning_reward: float,
    ):
        self.env: TimeLimit = gym.make(env_name, **env_kwargs)
        self.env.seed()
        self.env.action_space.np_random.seed(SEED)
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.episode_before_evaluation = episode_before_evaluation
        self.episode_maximum_size = episode_maximum_size
        self.minimum_accepted_score = minimum_accepted_score
        self.winning_reward = winning_reward


def evaluate(
    env: TimeLimit,
    total_episodes: int,
    *,
    q_table: np.ndarray = None,
    winning_reward: float = None,
    is_random: bool = False,
    render: bool = False,
    display_result: bool = False,
) -> float:
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
    :param display_result: If True, prints evaluation summary in the console at the evaluation end
    """
    # Todo : rename and re-think is_random parameter into policy parameter
    # Todo : render only last evaluation
    # Todo : yield q-table, evaluate it and continue evaluation if it is not good enough

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

    score = round(total_won_episodes * 100 / total_episodes, 2)

    if display_result:
        print("-" * 30)
        print(
            f"Results after {total_episodes} episodes using {'random' if is_random else 'q_table'}:"
        )
        print(f"Average steps per episode: {total_epochs / total_episodes}")
        print(f"Average reward per episode: {total_reward / total_episodes}")
        print(f"Percentage of won episodes : {score}%")
    return score


SEED = 0 or int(time())
random.seed(SEED)
