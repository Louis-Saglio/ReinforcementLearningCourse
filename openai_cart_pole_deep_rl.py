import pickle
from typing import Callable, Any, Tuple, List, Dict, Union

import gym
import numpy as np
from gym.wrappers import TimeLimit
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def play_one_session(
    env: TimeLimit,
    max_size: int,
    action_chooser: Callable[[TimeLimit, Any], Any],
    render: bool = False,
    custom_actions: Callable[[int, TimeLimit, Any, Any, Any, bool, Any], None] = None,
    stop_when_done: bool = True,
) -> Tuple[float, List[Dict[str, Any]]]:
    observation = env.reset()

    score = 0
    history = []

    for i in range(max_size):

        action = action_chooser(env, observation)
        current_iteration_history = {"observation": observation, "action": action}
        observation, reward, done, info = env.step(action)

        score += reward
        history.append(current_iteration_history)

        if custom_actions is not None:
            custom_actions(i, env, action, observation, reward, done, info)

        if render:
            env.render()

        if stop_when_done and done:
            break

    return score / max_size, history


def play_by_hand(env: TimeLimit):
    for _ in range(10):
        play_one_session(env, 10, lambda e, _: {"1": 1}.get(input(">>>"), 0), True)


def play_at_random(env: TimeLimit):
    def log(index, _, action, obs, reward, done, info):
        print(
            "-" * 30,
            f"Step: {index}",
            f"Action : {action}",
            f"Observation : {obs}",
            f"Reward : {reward}",
            f"Done : {done}",
            f"Info : {info}",
            sep="\n",
        )

    for _ in range(10):
        score, _ = play_one_session(
            env, 200, lambda e, _: e.action_space.sample(), False, log
        )
        print("Average score :", score)


def build_training_data(
    env: TimeLimit,
    min_training_data_length_wanted: int,
    training_duration: int,
    minimum_score: Union[int, float],
    action_chooser: Callable[[TimeLimit, Any], Any],
    show_progress: bool = False,
):
    training_data = []
    while len(training_data) < min_training_data_length_wanted:
        score, history = play_one_session(env, training_duration, action_chooser)

        if score >= minimum_score:
            for data in history:
                if data["action"] == 1:
                    action = [0, 1]
                elif data["action"] == 0:
                    action = [1, 0]
                else:
                    raise RuntimeError(f"Unexpected action value {data['action']}")
                training_data.append([data["observation"], action])

        if show_progress:
            print(
                f"\r{round(len(training_data) * 100 / min_training_data_length_wanted, 2)} %",
                end="",
            )
    if show_progress:
        print()
    return training_data


def build_training_data_by_random(
    env: TimeLimit,
    min_training_data_length_wanted: int,
    training_duration: int,
    minimum_score: Union[int, float],
    show_progress: bool = False,
):
    return build_training_data(
        env,
        min_training_data_length_wanted,
        training_duration,
        minimum_score,
        lambda e, _: e.action_space.sample(),
        show_progress,
    )


def build_training_data_with_model(
    env: TimeLimit,
    model: Sequential,
    min_training_data_length_wanted: int,
    training_duration: int,
    minimum_score: Union[int, float],
    show_progress: bool = False,
):
    def choose_smart_action(_: TimeLimit, observation):
        return np.argmax(model.predict(observation.reshape(-1, len(observation)))[0])

    return build_training_data(
        env,
        min_training_data_length_wanted,
        training_duration,
        minimum_score,
        choose_smart_action,
        show_progress,
    )


def build_model(input_size, output_size) -> Sequential:
    model = Sequential()
    model.add(Dense(128, input_dim=input_size, activation="relu"))
    model.add(Dense(52, activation="relu"))
    model.add(Dense(32, activation="sigmoid"))
    model.add(Dense(20, activation="sigmoid"))
    model.add(Dense(output_size, activation="linear"))
    model.compile(loss="mse", optimizer=Adam())
    return model


def train_model(training_data, dump_into: str = None) -> Sequential:
    x = np.array([observation for observation, _ in training_data])
    y = np.array([action for _, action in training_data])
    model = build_model(input_size=len(x[0]), output_size=len(y[0]))
    model.fit(x, y, epochs=10, verbose=0)
    if dump_into is not None:
        with open(dump_into, "wb") as f:
            pickle.dump(model, f)
    return model


def play_smart(
    env: TimeLimit, model: Sequential, session_numbers: int, session_size: int
):
    def choose_smart_action(_: TimeLimit, observation):
        # Todo : don't understand argmax
        return np.argmax(model.predict(observation.reshape(-1, len(observation)))[0])

    for _ in range(session_numbers):
        score, _ = play_one_session(
            env, session_size, choose_smart_action, True, stop_when_done=True
        )
        print(f"Average score : {round(score * 100, 2)} %")


def main(
    load_training_data_from: str = None,
    dump_training_data_in: str = None,
    load_model_from: str = None,
    dump_model_in: str = None,
):
    env = gym.make("CartPole-v1")

    # play_at_random(env)
    # play_by_hand(env)

    if load_model_from is not None:
        # If a model file is given, load it now
        # so we can build more qualified training data with it
        with open(load_model_from, "rb") as f:
            model = pickle.load(f)
        # Do not add an else clause for building & training a new model right now
        # because we must build training data before

    if load_training_data_from is not None:
        # If a training data file is given, use it
        with open(load_training_data_from, "rb") as f:
            training_data = pickle.load(f)
    elif load_model_from is not None:
        # Else if a model file is given, build training data with it
        training_data = build_training_data_with_model(
            env=env,
            model=model,
            min_training_data_length_wanted=100_000,
            training_duration=500,
            minimum_score=1,
            show_progress=True,
        )
    else:
        # Else build training data from random
        training_data = build_training_data_by_random(
            env,
            min_training_data_length_wanted=1000,
            training_duration=500,
            minimum_score=0.27,
            show_progress=True,
        )

    if load_model_from is None:
        # If not model file has been given, no model has been built yet
        # so build one now
        model = train_model(training_data)

    # Actually play the game
    play_smart(env, model, session_numbers=10, session_size=500)

    if (
        dump_training_data_in is not None
        and dump_training_data_in != load_training_data_from
    ):
        # If saving training data to file has been asked
        # and if the training data file is not the same that the file where to save them
        # (to prevent saving data to a file where they already are)
        with open(dump_training_data_in, "wb") as f:
            pickle.dump(training_data, f)

    if dump_model_in is not None and dump_model_in != load_model_from:
        # Same but with the model
        with open(dump_model_in, "wb") as f:
            pickle.dump(model, f)


if __name__ == "__main__":
    main(
        load_training_data_from="cart_pole_training_data.pickle",
        dump_training_data_in="cart_pole_training_data.pickle",
        load_model_from="cart_pole_model.pickle",
        dump_model_in="cart_pole_model.pickle",
    )
