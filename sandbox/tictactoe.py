from abc import ABC, abstractmethod
from collections import defaultdict
import math
from math import sqrt, log


class Node(ABC):
    """
    Noeud représentant l'état du jeu à un instant t.
    MCTS fonctionne en explorant un arbre de ces noeuds.
    Cela peut s'appliquer à un jeu d'echecs, de dames, de go ou autre.
    """

    @abstractmethod
    def find_children(self):
        """Retourne tous les enfants de ce noeud"""
        return set()

    @abstractmethod
    def find_random_child(self):
        """Retourne un enfant aléatoire du noeud courant (facilite la simulation)"""
        return None

    @abstractmethod
    def is_terminal(self):
        """Retourne vrai si le noeud n'a pas d'enfant"""
        return True

    @abstractmethod
    def reward(self):
        """Retourne la récompense de ce noeud 1=win, 0=loss, .5=tie, etc"""
        return 0

    @abstractmethod
    def __hash__(self):
        """Retourne un hash du noeud"""
        return 123456789

    @abstractmethod
    def __eq__(self, node2):
        """Fonction pour comparer deux noeuds"""
        return True


class MCTS:
    """Implémentation de l'algorithme Monte Carlo Tree Search"""

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # nombre total de reward pour chaque noeud
        self.N = defaultdict(int)  # nombre total de visites pour chaque noeud
        self.children = dict()  # enfants de chaque noeud
        self.exploration_weight = exploration_weight

    def choose(self, node):
        """Retourne le meilleur successeur de chaque noeud (la meilleure action dans le jeu)"""
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(child):
            # done
            return self.N[child] / self.Q[child] * sqrt(2) * (log(self.N[node]) / self.Q[node])

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        """Fait une passe d'entrainement sur l'arbre à partir d'un noeud"""
        path = self._select(node)
        leaf = path[-1]
        # TODO

    def _select(self, node):
        """Retourne le chemin vers un noeud descendant inexploré"""
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        """Met à jour la liste des enfants du noeud avec les enfants trouvés"""
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node):
        """Retourne la récompense d'une simulation (complete) du noeud"""
        invert_reward = True
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward

    def _backpropagate(self, path, reward):
        """Remonte la récompense aux ancetres du noeud"""
        for node in reversed(path):
            # TODO
            pass

    def _uct_select(self, node):
        """Selection un noeud enfant a explorer, selon l'algo"""

        # Tous les enfants du noeud doivent déjà etre étendus
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            # TODO

        return max(self.children[node], key=uct)


from collections import namedtuple
from random import choice

_TTTB = namedtuple("TicTacToeBoard", "tup turn winner terminal")


# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class TicTacToeBoard(_TTTB, Node):
    def find_children(self):
        if self.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in each of the empty spots
        return {self.make_move(i) for i, value in enumerate(self.tup) if value is None}

    def find_random_child(self):
        if self.terminal:
            return None  # If the game is finished then no moves can be made
        empty_spots = [i for i, value in enumerate(self.tup) if value is None]
        return self.make_move(choice(empty_spots))

    def reward(self):
        if not self.terminal:
            raise RuntimeError(f"reward called on nonterminal self {self}")
        if self.winner is self.turn:
            # It's your turn and you've already won. Should be impossible.
            raise RuntimeError(f"reward called on unreachable self {self}")
        if self.turn is (not self.winner):
            return 0  # Your opponent has just won. Bad.
        if self.winner is None:
            return 0.5  # self.is a tie
        # The winner is neither True, False, nor None
        raise RuntimeError(f"self has unknown winner type {self.winner}")

    def is_terminal(self):
        return self.terminal

    def make_move(self, index):
        tup = self.tup[:index] + (self.turn,) + self.tup[index + 1 :]
        turn = not self.turn
        winner = _find_winner(tup)
        is_terminal = (winner is not None) or not any(v is None for v in tup)
        return TicTacToeBoard(tup, turn, winner, is_terminal)

    def to_pretty_string(self):
        to_char = lambda v: ("X" if v is True else ("O" if v is False else " "))
        rows = [
            [to_char(self.tup[3 * row + col]) for col in range(3)] for row in range(3)
        ]
        return (
            "\n  1 2 3\n"
            + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
            + "\n"
        )


def play_game():
    tree = MCTS()
    board = new_tic_tac_toe_board()
    print(board.to_pretty_string())
    while True:
        row_col = input("enter row,col: ")
        row, col = map(int, row_col.split(","))
        index = 3 * (row - 1) + (col - 1)
        if board.tup[index] is not None:
            raise RuntimeError("Invalid move")
        board = board.make_move(index)
        print(board.to_pretty_string())
        if board.terminal:
            break
        # You can train as you go, or only at the beginning.
        # Here, we train as we go, doing fifty rollouts each turn.
        for _ in range(50):
            tree.do_rollout(board)
        board = tree.choose(board)
        print(board.to_pretty_string())
        if board.terminal:
            break


def _winning_combos():
    for start in range(0, 9, 3):  # three in a row
        yield start, start + 1, start + 2
    for start in range(3):  # three in a column
        yield start, start + 3, start + 6
    yield 0, 4, 8  # down-right diagonal
    yield 2, 4, 6  # down-left diagonal


def _find_winner(tup):
    """Returns None if no winner, True if X wins, False if O wins"""
    for i1, i2, i3 in _winning_combos():
        v1, v2, v3 = tup[i1], tup[i2], tup[i3]
        if False is v1 is v2 is v3:
            return False
        if True is v1 is v2 is v3:
            return True
    return None


def new_tic_tac_toe_board():
    return TicTacToeBoard(tup=(None,) * 9, turn=True, winner=None, terminal=False)


if __name__ == "__main__":
    play_game()
