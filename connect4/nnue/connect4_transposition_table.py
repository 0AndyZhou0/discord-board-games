import numpy as np

from .connect4 import Connect4


class Connect4TranspositionTable:
    def __init__(self) -> None:
        self.table = {}

    def add(self, red_bitboard: np.longlong, yellow_bitboard: np.longlong, value: np.double) -> None:
        self.table[(red_bitboard, yellow_bitboard)] = value

    def get(self, red_bitboard: np.longlong, yellow_bitboard: np.longlong) -> np.double:
        return self.table.get((red_bitboard, yellow_bitboard), np.double(0))

    def contains(self, red_bitboard: np.longlong, yellow_bitboard: np.longlong) -> bool:
        return (red_bitboard, yellow_bitboard) in self.table

    def clear(self) -> None:
        self.table = {}