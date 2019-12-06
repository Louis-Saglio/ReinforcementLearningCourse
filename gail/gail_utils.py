from typing import Callable, Any, Tuple, List, Dict, Union

import numpy as np
from gym.wrappers import TimeLimit
from keras import Sequential


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

        if render:
            env.render()

        action = action_chooser(env, observation)
        current_iteration_history = {"observation": observation, "action": action}
        observation, reward, done, info = env.step(action.reshape((-1,)))

        score += reward
        history.append(current_iteration_history)

        if custom_actions is not None:
            custom_actions(i, env, action, observation, reward, done, info)

        if stop_when_done and done:
            break

    return score / max_size, history


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
        score, history = play_one_session(
            env, training_duration, action_chooser, render=False
        )
        print(score)

        if score >= minimum_score:
            for data in history:
                #     if data["action"] == 1:
                #         action = [0, 1]
                #     elif data["action"] == 0:
                #         action = [1, 0]
                #     else:
                #         raise RuntimeError(f"Unexpected action value {data['action']}")
                training_data.append([data["observation"], data["action"]])

        if show_progress:
            print(
                f"\r{round(len(training_data) * 100 / min_training_data_length_wanted, 2)} %",
                end="",
            )
    if show_progress:
        print()
    return training_data
