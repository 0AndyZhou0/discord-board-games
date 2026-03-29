import logging

import numpy as np

from .connect4 import Color, Connect4

logger = logging.getLogger("cogs.connect4.nnue")

class Connect4Game:
    def __init__(self) -> None:
        self.player = Color.RED
        self.red_bitboard = np.longlong(0)
        self.yellow_bitboard = np.longlong(0)
        self.features = np.zeros(Connect4.rows * Connect4.cols * 2) # idx(r, c, player) = (rows*cols)*player + rows*c + r

    def print_bitboard(self) -> None:
        for i in range(Connect4.rows):
            for j in range(Connect4.cols):
                if self.red_bitboard & (1 << (i * Connect4.cols + j)):
                    print("R", end="")
                elif self.yellow_bitboard & (1 << (i * Connect4.cols + j)):
                    print("Y", end="")
                else:
                    print(".", end="")
            print()

    def drop_piece(self, col: int, player: Color) -> None:
        r = None
        c = col
        for row in range(Connect4.rows - 1, -1, -1):
            if not self.red_bitboard & (1 << (row * Connect4.cols + col)) and not self.yellow_bitboard & (1 << (row * Connect4.cols + col)):
                r = row
                break
        if r == Connect4.rows:
            logger.error(f"Column {col} is full")
            return
        match player:
            case Color.RED:
                self.red_bitboard |= 1 << (r * Connect4.cols + c)
            case Color.YELLOW:
                self.yellow_bitboard |= 1 << (r * Connect4.cols + c)
        self.add_feature(r, c, player)
        # Update NNUE

    def add_feature(self, row: int, col: int, player: int, value: int = 1) -> None:
        """sets the feature for the given row, col, player to the given value (0 or 1)

        Args:
            row (int)
            col (int)
            player (int): 
        """
        idx = Connect4.rows * Connect4.cols * player + row * Connect4.cols + col
        self.features[idx] = value