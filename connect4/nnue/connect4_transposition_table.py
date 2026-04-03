import numpy as np

from .connect4 import Connect4


class Connect4TranspositionTable:
    def __init__(self) -> None:
        self.table = {}

    def add(self, red_bitboard: np.longlong, yellow_bitboard: np.longlong, evaluations: np.array) -> None:
        self.table[(red_bitboard, yellow_bitboard)] = evaluations
        
    def get(self, red_bitboard: np.longlong, yellow_bitboard: np.longlong) -> np.array:
        return self.table.get((red_bitboard, yellow_bitboard), np.zeros(Connect4.cols))
    
    def contains(self, red_bitboard: np.longlong, yellow_bitboard: np.longlong) -> bool:
        return (red_bitboard, yellow_bitboard) in self.table
    
    def clear(self) -> None:
        self.table = {}