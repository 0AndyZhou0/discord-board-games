from pathlib import Path
import time

import numpy as np

from .external_functions import get_eval

from .connect4 import Connect4
from .connect4_color import Color
from .connect4_game import Connect4Game
from .connect4_transposition_table import Connect4TranspositionTable


class Connect4Minimax:
    def __init__(self) -> None:
        self.table = Connect4TranspositionTable()
        self.total_terminal_time = 0
        self.total_non_terminal_time = 0

    def get_winning_squares(self, game: Connect4Game, player: Color) -> list[tuple[int, int]]:
        """Returns a list of squares that would result in a win for the given player"""
        winning_squares = []
        for row in range(Connect4.rows):
            for col in range(Connect4.cols):
                if game.is_column_full(col):
                    continue
                if game.red_bitboard & (1 << (row * Connect4.cols + col)) or game.yellow_bitboard & (1 << (row * Connect4.cols + col)):
                    continue
                if player == Color.RED:
                    game.red_bitboard |= 1 << (row * Connect4.cols + col)
                    if game.get_winner():
                        winning_squares.append((row, col))
                    game.red_bitboard &= ~(1 << (row * Connect4.cols + col))
                else:
                    game.yellow_bitboard |= 1 << (row * Connect4.cols + col)
                    if game.get_winner():
                        winning_squares.append((row, col))
                    game.yellow_bitboard &= ~(1 << (row * Connect4.cols + col))
        return winning_squares

    def minimax(self, game: Connect4Game, prev_move: tuple[int, int], depth: int, alpha: float = -np.inf, beta: float = np.inf) -> float:
        """Returns value of the board for the given player"""
        # Terminal Node
        winner = game.get_winner_from_move(*prev_move)
        if depth <= 0 or winner is not None:
            start_time = time.time()
            if winner is not None:
                self.total_terminal_time += time.time() - start_time
                return -1000.0 - depth
            self.total_terminal_time += time.time() - start_time
            # return get_eval(game.moves)
            return game.nnue_wrapper.accumulator_forward(game.player)
            
        # TODO: Implement check for fastest win and prune

        # TODO: Implement null move pruning
            

        start_time = time.time()
        best_value = -np.inf
        evals = self.table.get(game.red_bitboard, game.yellow_bitboard)
        order = np.flip(np.argsort(evals))
        new_order = np.zeros(Connect4.cols)
        for col in order:
            if game.is_column_full(col):
                new_order[col] = -np.inf
                continue
            row, col = game.drop_piece(col)
            self.total_non_terminal_time += time.time() - start_time
            value = -self.minimax(game, (row, col), depth - 1, -beta, -alpha)
            start_time = time.time()
            game.remove_piece(row, col)
            new_order[col] = value
            if value > best_value:
                best_value = value
                alpha = max(alpha, best_value)
            if value >= beta:
                return best_value

        self.table.add(game.red_bitboard, game.yellow_bitboard, new_order)
        self.total_non_terminal_time += time.time() - start_time
        return best_value
    
    def iterative_deepening(self, game: Connect4Game, depth: int = 5) -> int:
        game.evaluate_board_reset()
        for d in range(depth):
            col = self.get_best_col(game, d)
        return col
    
    def get_best_col(self, game: Connect4Game, depth: int) -> int:
        best_col = None
        best_value = -np.inf
        # Get column order
        evals = self.table.get(game.red_bitboard, game.yellow_bitboard)
        order = np.flip(np.argsort(evals))
        new_order = np.zeros(Connect4.cols)
        for col in order:
            if game.is_column_full(col):
                new_order[col] = -np.inf
                continue
            row, col = game.drop_piece(col)
            value = -self.minimax(game, (row, col), depth)
            game.remove_piece(row, col)
            new_order[col] = value
            if value > best_value:
                best_value = value
                best_col = col
        self.table.add(game.red_bitboard, game.yellow_bitboard, new_order)
        # print("New Order: ", new_order)
        assert best_col is not None
        return best_col