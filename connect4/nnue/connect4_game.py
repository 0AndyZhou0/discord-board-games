import logging

import numpy as np

from .connect4_nnue_wrapper import Connect4NNUEWrapper

from .connect4 import Connect4
from .connect4_color import Color
from .connect4_nnue import Connect4NNUE

logger = logging.getLogger("cogs.connect4.nnue")

class Connect4Game:
    def __init__(self) -> None:
        self.player: Color = Color.RED
        self.moves: str = ""
        self.red_bitboard: np.longlong = np.longlong(0)
        self.yellow_bitboard: np.longlong = np.longlong(0)

        """
        bitboard layout:\n
        0  1  2  3  4  5  6\n
        7  8  9  10 11 12 13\n
        14 15 16 17 18 19 20\n
        21 22 23 24 25 26 27\n
        28 29 30 31 32 33 34\n
        35 36 37 38 39 40 41
        """
        # self.features = np.zeros(Connect4.rows * Connect4.cols * 2) # idx(r, c, player) = (rows*cols)*player + rows*c + r
        # self.nnue = Connect4NNUE()
        self.nnue_wrapper = Connect4NNUEWrapper()
        self.nnue_wrapper.evaluate_board(self.red_bitboard, self.yellow_bitboard, self.player)

    def load_model(self, path: str) -> None:
        self.nnue_wrapper.load_model(path)

    def reset(self) -> None:
        self.red_bitboard = np.longlong(0)
        self.yellow_bitboard = np.longlong(0)
        self.player = Color.RED
        self.moves = ""
        self.evaluate_board_reset()

    def evaluate_board_reset(self) -> float:
        return self.nnue_wrapper.evaluate_board(self.red_bitboard, self.yellow_bitboard, self.player)
    
    def evaluate_board(self) -> float:
        return self.nnue_wrapper.accumulator_forward(self.player)

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

    def to_string(self) -> None:
        s = ""
        for i in range(Connect4.rows):
            for j in range(Connect4.cols):
                if self.red_bitboard & (1 << (i * Connect4.cols + j)):
                    s += "R"
                elif self.yellow_bitboard & (1 << (i * Connect4.cols + j)):
                    s += "Y"
                else:
                    s += "."
            s += "\n"
        return s

    def is_column_full(self, col: int) -> bool:
        return self.red_bitboard & (1 << col) or self.yellow_bitboard & (1 << col)
    
    def get_valid_cols(self) -> list[int]:
        valid_cols = []
        for col in [3, 2, 4, 1, 5, 0, 6]: # middle columns first for better move ordering
            if not self.is_column_full(col):
                valid_cols.append(col)
        return valid_cols
    
    def get_valid_cols_mask(self) -> np.array:
        valid_cols = np.zeros(Connect4.cols)
        for col in range(Connect4.cols):
            if not self.is_column_full(col):
                valid_cols[col] = 1
        return valid_cols

    def drop_piece_with_color(self, col: int, player: Color) -> tuple[int, int]:
        raise DeprecationWarning("Use drop_piece instead")
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
        self.player *= -1
        self.moves += str(col+1)
        return r, c
    
    def get_bitboards_from_drop(self, col: int) -> tuple[np.longlong, np.longlong]:
        r = None
        c = col
        for row in range(Connect4.rows - 1, -1, -1):
            if not self.red_bitboard & (1 << (row * Connect4.cols + col)) and not self.yellow_bitboard & (1 << (row * Connect4.cols + col)):
                r = row
                break
        if r is None:
            logger.error(f"Column {col} is full")
            raise Exception("Column is full")
        match self.player:
            case Color.RED:
                new_red_bitboard = self.red_bitboard | (1 << (r * Connect4.cols + c))
                return new_red_bitboard, self.yellow_bitboard
            case Color.YELLOW:
                new_yellow_bitboard = self.yellow_bitboard | (1 << (r * Connect4.cols + c))
                return self.red_bitboard, new_yellow_bitboard
        raise Exception("Player not found")
    
    def drop_piece(self, col: int) -> tuple[int, int]:
        r = None
        c = col
        for row in range(Connect4.rows - 1, -1, -1):
            if not self.red_bitboard & (1 << (row * Connect4.cols + col)) and not self.yellow_bitboard & (1 << (row * Connect4.cols + col)):
                r = row
                break
        if r is None:
            logger.error(f"Column {col} is full")
            raise Exception("Column is full")
        match self.player:
            case Color.RED:
                self.red_bitboard |= 1 << (r * Connect4.cols + c)
            case Color.YELLOW:
                self.yellow_bitboard |= 1 << (r * Connect4.cols + c)
        # Update NNUE
        self.add_feature(r, c, self.player)
        self.player *= -1
        self.moves += str(col+1)
        return r, c
    
    def remove_piece_by_color(self, row: int, col: int, player: Color) -> None:
        raise DeprecationWarning("Use remove_piece instead")
        match player:
            case Color.RED:
                self.red_bitboard &= ~(1 << (row * Connect4.cols + col))
            case Color.YELLOW:
                self.yellow_bitboard &= ~(1 << (row * Connect4.cols + col))
        # Update NNUE
        self.remove_feature(row, col, player)
        self.moves = self.moves[:-1]
        self.player *= -1

    def remove_piece(self, row: int, col: int) -> None:
        player = -self.player
        match player:
            case Color.RED:
                self.red_bitboard &= ~(1 << (row * Connect4.cols + col))
            case Color.YELLOW:
                self.yellow_bitboard &= ~(1 << (row * Connect4.cols + col))
        # Update NNUE
        self.remove_feature(row, col, player)
        self.moves = self.moves[:-1]
        self.player *= -1

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
    
    def get_winner_from_move(self, row: int, col: int) -> Color | None:
        """Returns -1 if red wins, 1 if yellow wins, 1e-4 if tie, and None if no winner yet"""
        color = Color.RED if self.red_bitboard & (1 << (row * Connect4.cols + col)) else Color.YELLOW
        bitboard = self.red_bitboard if color == Color.RED else self.yellow_bitboard
        # Vertical
        count = 0
        for c in range(Connect4.cols):
            if bitboard & (1 << (row * Connect4.cols + c)):
                count += 1
            else:
                count = 0
            if count == 4:
                return color
        
        # Horizontal
        count = 0
        for r in range(Connect4.rows):
            if bitboard & (1 << (r * Connect4.cols + col)):
                count += 1
            else:
                count = 0
            if count == 4:
                return color
        
        # Diagonal \
        count = 0
        top_left_row = row - min(row, col)
        top_left_col = col - min(row, col)
        while top_left_row < Connect4.rows and top_left_col < Connect4.cols:
            if bitboard & (1 << (top_left_row * Connect4.cols + top_left_col)):
                count += 1
            else:
                count = 0
            if count == 4:
                return color
            top_left_row += 1
            top_left_col += 1
        
        # Diagonal /
        count = 0
        bottom_left_row = row + min(Connect4.rows - 1 - row, col)
        bottom_left_col = col - min(Connect4.rows - 1 - row, col)
        while bottom_left_row >= 0 and bottom_left_col < Connect4.cols:
            if bitboard & (1 << (bottom_left_row * Connect4.cols + bottom_left_col)):
                count += 1
            else:
                count = 0
            if count == 4:
                return color
            bottom_left_row -= 1
            bottom_left_col += 1

        if self.red_bitboard | self.yellow_bitboard == (1 << (Connect4.rows * Connect4.cols)) - 1:
            return 1e-4 # tie
        
        return None

    def get_winner(self) -> Color | None:
        """Returns -1 if red wins, 1 if yellow wins, 1e-4 if tie, and None if no winner yet"""
        y = self.red_bitboard & (self.red_bitboard >> 7) # diagonal /
        if (y & (y >> 2 * 6)): 
            print("diagonal /")
            return Color.RED
        y = self.red_bitboard & (self.red_bitboard >> 7) # vertical
        if (y & (y >> 2 * 7)): 
            print("vertical")
            return Color.RED
        y = self.red_bitboard & (self.red_bitboard >> 8) # diagonal \
        if (y & (y >> 2 * 8)): 
            print("diagonal \\")
            return Color.RED
        y = self.red_bitboard & (self.red_bitboard >> 1) # horizontal
        if (y & (y >> 2)): 
            print("horizontal")
            return Color.RED
        
        y = self.yellow_bitboard & (self.yellow_bitboard >> 8) # diagonal /
        if (y & (y >> 2 * 8)): 
            print("diagonal /")
            return Color.YELLOW
        y = self.yellow_bitboard & (self.yellow_bitboard >> 7) # vertical
        if (y & (y >> 2 * 7)): 
            print("vertical")
            return Color.YELLOW
        y = self.yellow_bitboard & (self.yellow_bitboard >> 9) # diagonal \
        if (y & (y >> 2 * 9)): 
            self.red_bitboard = 0
            # self.yellow_bitboard = self.yellow_bitboard >> 8
            # self.yellow_bitboard = y
            self.yellow_bitboard = y >> 2 * 5
            print()
            self.print_bitboard()
            exit(0)
            print("diagonal \\")
            return Color.YELLOW
        y = self.yellow_bitboard & (self.yellow_bitboard >> 1) # horizontal
        if (y & (y >> 2)): 
            print("horizontal")
            return Color.YELLOW
        
        if self.red_bitboard | self.yellow_bitboard == (1 << (Connect4.rows * Connect4.cols)) - 1:
            return 1e-4 # tie
        
        return None;