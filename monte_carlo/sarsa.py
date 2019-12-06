import random
from time import time

import numpy as np

from utils import evaluate, Problem


def choose_action(env, q_table, from_state, epsilon: float):
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[from_state, :])
    return action


def train(
    problem: Problem, q_table=None,
):
    """
    Build a Q-table, which can be later used to solve the problem given by env
    :param problem:
    :param q_table: Start with this q-table. If None, will build a zero filled q-table
    :return: A Q-table trained to solve the problem given by env
    """

    if q_table is None:
        q_table = np.zeros(
            (problem.env.observation_space.n, problem.env.action_space.n)
        )

    while True:
        for episode_index in range(problem.episode_before_evaluation):
            previous_state = problem.env.reset()
            previous_action = choose_action(
                problem.env, q_table, previous_state, problem.epsilon
            )
            for step_index in range(problem.episode_maximum_size):

                current_state, current_state_reward, done, info = problem.env.step(
                    previous_action
                )

                next_action = choose_action(
                    problem.env, q_table, current_state, problem.epsilon
                )

                q_table[previous_state, previous_action] += problem.learning_rate * (
                    current_state_reward
                    + problem.discount_factor * q_table[current_state, next_action]
                    - q_table[previous_state, previous_action]
                )

                previous_state = current_state
                previous_action = next_action

                if done:
                    break

        yield q_table


def main():
    frozen_lake = Problem(
        env_name="FrozenLake-v0",
        env_kwargs={"is_slippery": False},
        epsilon=0.3,
        learning_rate=0.8,
        discount_factor=0.95,
        episode_before_evaluation=100,
        episode_maximum_size=300,
        minimum_accepted_score=100,
        winning_reward=1,
    )
    frozen_lake_8x8 = Problem(
        env_name="FrozenLake8x8-v0",
        env_kwargs={"is_slippery": False},
        epsilon=0.3,
        learning_rate=0.8,
        discount_factor=0.95,
        episode_before_evaluation=100,
        episode_maximum_size=300,
        minimum_accepted_score=100,
        winning_reward=1,
    )
    taxi = Problem(
        env_name="Taxi-v3",
        env_kwargs={},
        epsilon=0.0,
        learning_rate=0.8,
        discount_factor=0.95,
        episode_before_evaluation=100,
        episode_maximum_size=300,
        minimum_accepted_score=100,
        winning_reward=20,
    )

    problem = taxi

    q_table_generator = train(problem)
    q_table = next(q_table_generator)

    not_solved = True
    while not_solved:
        try:
            q_table = next(q_table_generator)
            not_solved = (
                evaluate(
                    problem.env,
                    100,
                    q_table=q_table,
                    winning_reward=problem.winning_reward,
                    render=False,
                    display_result=True,
                )
                < problem.minimum_accepted_score
            )
        except KeyboardInterrupt:
            breakpoint()

    evaluate(
        problem.env,
        100,
        is_random=True,
        winning_reward=problem.winning_reward,
        display_result=True,
    )

    return problem, q_table


if __name__ == "__main__":
    environment, Q = main()
