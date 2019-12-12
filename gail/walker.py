import pickle
from statistics import mean
from typing import Any, List, Union, Callable

import gym
import numpy as np
from gym.wrappers import TimeLimit
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from gail_utils import play_one_session


class EpisodeResume:
    def __init__(self, reward, memory):
        self.memory = memory
        self.reward = reward


def build_training_data(
    env: TimeLimit,
    min_training_data_length_wanted: int,
    training_duration: int,
    minimum_score: Union[int, float],
    action_chooser: Callable[[TimeLimit, Any], Any],
    show_progress: bool = False,
):
    training_data = []
    scores = []
    while len(training_data) < min_training_data_length_wanted:
        score, history = play_one_session(
            env, training_duration, action_chooser, render=False
        )
        scores.append(score)

        if score >= minimum_score:
            for data in history:
                training_data.append([data["observation"], data["action"]])

        if show_progress:
            print(
                f"\r{round(len(training_data) * 100 / min_training_data_length_wanted, 2)} %",
                end="",
            )
    if show_progress:
        print()
    return (
        (np.array([observation for observation, _ in training_data])),
        (np.array([action for _, action in training_data])),
        scores,
    )


def replay_memory(env: TimeLimit, memory: List[List[Any]]):
    for episode_memory in memory:
        env.reset()
        for action in episode_memory:
            env.step(action)
            env.render()


def build_model(input_size, output_size) -> Sequential:
    model = Sequential()
    model.add(Dense(64, input_dim=input_size, activation="relu"))
    model.add(Dense(output_size, activation="sigmoid"))
    model.compile(loss="mse", optimizer=Adam())
    return model


def play_smart(
    env: TimeLimit, model: Sequential, session_numbers: int, session_size: int
):
    def choose_smart_action(_: TimeLimit, observation):
        return model.predict(np.array([observation]))

    scores = []
    for _ in range(session_numbers):
        score, _ = play_one_session(
            env, session_size, choose_smart_action, render=False, stop_when_done=True
        )
        scores.append(score)
    print(f"Average score : {round(mean(scores), 2)}")


def play_at_random(env: TimeLimit, session_numbers: int, session_size: int):
    scores = []
    for _ in range(session_numbers):
        score, _ = play_one_session(
            env,
            session_size,
            lambda e, _: e.action_space.sample(),
            render=False,
            stop_when_done=True,
        )
        scores.append(score)
    print(f"Average score : {round(mean(scores), 2)}")


def main(load_from=None):
    env = gym.make("BipedalWalker-v2")
    if load_from:
        with open(load_from, "rb") as f:
            states, actions, scores = pickle.load(f)
    else:
        states, actions, scores = build_training_data(
            env,
            min_training_data_length_wanted=10000,
            training_duration=1600,
            minimum_score=-0.05,
            action_chooser=lambda e, _: e.action_space.sample(),
            show_progress=True,
        )
        with open("training", "wb") as f:
            pickle.dump((states, actions, scores), f)

    model = build_model(input_size=len(states[0]), output_size=len(actions[0]))
    model.fit(states, actions, epochs=1, verbose=1)

    print("play with model")
    play_smart(env, model, 100, 100)
    print("play at random")
    play_at_random(env, 100, 100)
    return locals()


if __name__ == "__main__":
    scope = main("training")
