from typing import Tuple, List


class Action:
    def __init__(self, from_state, to_states: List[Tuple["State", float]]):
        assert sum([to[0] for to in to_states]) == 1
        self.from_state = from_state
        self.to_states = to_states


class State:
    def __init__(self, name, reward: float):
        self.name = name
        self.reward = reward


class Environment:
    def __init__(self):
        self.states = [State("S0", 0), State("S1", 0), State("S2", 5), State("S3", -5)]
        self.states_as_dict = {state.name: state for state in self.states}
        self.actions = [
            Action(
                self.states_as_dict["S0"],
                [(self.states_as_dict["S0"], 0.2), (self.states_as_dict["S1"], 0.8)],
            ),
            Action(self.states_as_dict["S0"], [(self.states_as_dict["S0"], 1)]),
            Action(
                self.states_as_dict["S1"],
                [(self.states_as_dict["S1"], 0.2), (self.states_as_dict["S2"], 0.8)],
            ),
            Action(
                self.states_as_dict["S1"],
                [(self.states_as_dict["S1"], 0.2), (self.states_as_dict["S0"], 0.8)],
            ),
            Action(self.states_as_dict["S2"], [(self.states_as_dict["S3"], 1)]),
            Action(self.states_as_dict["S2"], [(self.states_as_dict["S1"], 1)]),
            Action(
                self.states_as_dict["S3"],
                [(self.states_as_dict["S0"], 0.8), (self.states_as_dict["S3"], 0.2)],
            ),
            Action(
                self.states_as_dict["S3"],
                [(self.states_as_dict["S2"], 0.9), (self.states_as_dict["S4"], 0.1)],
            ),
        ]


def main():
    env = Environment()
