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

    return score, history


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
):
    training_data = []
    while len(training_data) < min_training_data_length_wanted:
        score, history = play_one_session(
            env, training_duration, lambda e, _: e.action_space.sample()
        )

        if score >= minimum_score:
            for data in history:
                if data["action"] == 1:
                    action = [1]
                elif data["action"] == 0:
                    action = [0]
                else:
                    raise RuntimeError(f"Unexpected action value {data['action']}")
                training_data.append([data["observation"], action])
            # print(
            #     f"\r{round(len(training_data) * 100 / min_training_data_length_wanted, 2)} %",
            #     end="",
            # )
    return training_data


def build_model(input_size, output_size) -> Sequential:
    model = Sequential()
    model.add(Dense(128, input_dim=input_size, activation="relu"))
    model.add(Dense(52, activation="relu"))
    model.add(Dense(output_size, activation="linear"))
    model.compile(loss="mse", optimizer=Adam())
    return model


def train_model(training_data) -> Sequential:
    x = np.array([observation for observation, _ in training_data])
    y = np.array([action for _, action in training_data])
    model = build_model(input_size=len(x[0]), output_size=len(y[0]))
    model.fit(x, y, epochs=10, verbose=1)
    return model


def play_smart(
    env: TimeLimit, model: Sequential, session_numbers: int, session_size: int
):
    def choose_smart_action(_: TimeLimit, observation):
        # Todo : don't understand argmax
        return np.argmax(model.predict(observation.reshape(-1, len(observation)))[0])

    for _ in range(session_numbers):
        score, _ = play_one_session(env, session_size, choose_smart_action, True)
        print(f"Average score : {score}")


def main():
    env = gym.make("CartPole-v1")
    # play_at_random(env)
    # play_by_hand(env)
    training_data = build_training_data(
        env,
        min_training_data_length_wanted=1000,
        training_duration=100,
        minimum_score=100,
    )
    model = train_model(training_data)
    play_smart(env, model, session_numbers=10, session_size=100)


if __name__ == "__main__":
    main()
