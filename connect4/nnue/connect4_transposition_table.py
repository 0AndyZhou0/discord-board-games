import numpy as np

from .connect4 import Connect4


class Connect4TranspositionTable:
    def __init__(self) -> None:
        self.table = {}

    def add(self, moves: str, value: np.double) -> None:
        self.table[moves] = value

    def get(self, moves: str) -> np.double:
        return self.table.get(moves, np.double(0))

    def contains(self, moves: str) -> bool:
        return moves in self.table

    def clear(self) -> None:
        self.table = {}