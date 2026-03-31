import logging

import numpy as np

from .connect4_nnue_wrapper import Connect4NNUEWrapper

from .connect4 import Connect4
from .connect4_color import Color
from .connect4_nnue import Connect4NNUE

logger = logging.getLogger("cogs.connect4.nnue")

class Connect4Game:
    def __init__(self) -> None:
        self.player = Color.RED
        self.red_bitboard: np.longlong = np.longlong(0)
        self.yellow_bitboard: np.longlong = np.longlong(0)
        """
        bitboard layout:
        0  1  2  3  4  5  6
        7  8  9  10 11 12 13
        14 15 16 17 18 19 20
        21 22 23 24 25 26 27
        28 29 30 31 32 33 34
        35 36 37 38 39 40 41
        """
        # self.features = np.zeros(Connect4.rows * Connect4.cols * 2) # idx(r, c, player) = (rows*cols)*player + rows*c + r
        # self.nnue = Connect4NNUE()
        self.nnue_wrapper = Connect4NNUEWrapper()
        self.nnue_wrapper.evaluate_board(self.red_bitboard, self.yellow_bitboard, self.player)

    def load_model(self, path: str) -> None:
        self.nnue_wrapper.load_model(path)

    def evaluate_board(self) -> float:
        return self.nnue_wrapper.evaluate_board(self.red_bitboard, self.yellow_bitboard, self.player)

    def to_bitboards(self, board: np.array) -> tuple[np.longlong, np.longlong]:
        red_bitboard = np.longlong(0)
        yellow_bitboard = np.longlong(0)
        for r in range(Connect4.rows):
            for c in range(Connect4.cols):
                if board[r][c] == Color.RED:
                    red_bitboard |= 1 << (r * Connect4.cols + c)
                elif board[r][c] == Color.YELLOW:
                    yellow_bitboard |= 1 << (r * Connect4.cols + c)
        return red_bitboard, yellow_bitboard

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

    def is_column_full(self, col: int) -> bool:
        return self.red_bitboard & (1 << col) or self.yellow_bitboard & (1 << col)
    
    def get_valid_cols(self) -> list[int]:
        valid_cols = []
        for col in [3, 2, 4, 1, 5, 0, 6]: # middle columns first for better move ordering
            if not self.is_column_full(col):
                valid_cols.append(col)
        return valid_cols

    def drop_piece(self, col: int, player: Color) -> tuple[int, int]:
        r = None
        c = col
        for row in range(Connect4.rows - 1, -1, -1):
            if not self.red_bitboard & (1 << (row * Connect4.cols + col)) and not self.yellow_bitboard & (1 << (row * Connect4.cols + col)):
                r = row
                break
        if r is None:
            logger.error(f"Column {col} is full")
            raise Exception("Column is full")
        match player:
            case Color.RED:
                self.red_bitboard |= 1 << (r * Connect4.cols + c)
            case Color.YELLOW:
                self.yellow_bitboard |= 1 << (r * Connect4.cols + c)
        # Update NNUE
        self.add_feature(r, c, player)
        return r, c
    
    def remove_piece(self, row: int, col: int, player: Color) -> None:
        match player:
            case Color.RED:
                self.red_bitboard &= ~(1 << (row * Connect4.cols + col))
            case Color.YELLOW:
                self.yellow_bitboard &= ~(1 << (row * Connect4.cols + col))
        # Update NNUE
        self.remove_feature(row, col, player)

    def add_feature(self, row: int, col: int, player: int) -> None:
        """sets the feature for the given row, col, player"""
        self.nnue_wrapper.accumulator_add(row, col, player)

    def remove_feature(self, row: int, col: int, player: int) -> None:
        """unsets the feature for the given row, col, player"""
        self.nnue_wrapper.accumulator_remove(row, col, player)

    def check_for_win(self, player: Color) -> bool:
        bitboard = self.red_bitboard if player == Color.RED else self.yellow_bitboard
        y = bitboard & (bitboard >> 6) # diagonal /
        if (y & (y >> 2 * 6)): return True
        y = bitboard & (bitboard >> 7) # vertical
        if (y & (y >> 2 * 7)): return True
        y = bitboard & (bitboard >> 8) # diagonal \
        if (y & (y >> 2 * 8)): return True
        y = bitboard & (bitboard >> 1) # horizontal
        if (y & (y >> 2)): return True  # noqa: SIM103
        return False

    def get_winner(self) -> Color | None:
        """Returns -1 if red wins, 1 if yellow wins, 1e-4 if tie, and None if no winner yet"""
        y = self.red_bitboard & (self.red_bitboard >> 6) # diagonal /
        if (y & (y >> 2 * 6)): return Color.RED
        y = self.red_bitboard & (self.red_bitboard >> 7) # vertical
        if (y & (y >> 2 * 7)): return Color.RED
        y = self.red_bitboard & (self.red_bitboard >> 8) # diagonal \
        if (y & (y >> 2 * 8)): return Color.RED
        y = self.red_bitboard & (self.red_bitboard >> 1) # horizontal
        if (y & (y >> 2)): return Color.RED
        
        y = self.yellow_bitboard & (self.yellow_bitboard >> 6) # diagonal /
        if (y & (y >> 2 * 6)): return Color.YELLOW
        y = self.yellow_bitboard & (self.yellow_bitboard >> 7) # vertical
        if (y & (y >> 2 * 7)): return Color.YELLOW
        y = self.yellow_bitboard & (self.yellow_bitboard >> 8) # diagonal \
        if (y & (y >> 2 * 8)): return Color.YELLOW
        y = self.yellow_bitboard & (self.yellow_bitboard >> 1) # horizontal
        if (y & (y >> 2)): return Color.YELLOW
        
        if self.red_bitboard | self.yellow_bitboard == (1 << (Connect4.rows * Connect4.cols)) - 1:
            return 1e-4 # tie
        
        return None;