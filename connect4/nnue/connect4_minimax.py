from pathlib import Path
import time

import numpy as np

from .external_functions import get_eval

from .connect4 import Connect4
from .connect4_color import Color
from .connect4_game import Connect4Game


class Connect4Minimax:
    total_terminal_time = 0
    total_non_terminal_time = 0

    def get_winning_squares(game: Connect4Game, player: Color) -> list[tuple[int, int]]:
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

    def minimax(game: Connect4Game, prev_move: tuple[int, int], depth: int, alpha: float = -np.inf, beta: float = np.inf) -> float:
        """Returns value of the board for the given player"""
        # Terminal Node
        winner = game.get_winner_from_move(*prev_move)
        if depth <= 0 or winner is not None:
            start_time = time.time()
            if winner is not None:
                game.print_bitboard()
                Connect4Minimax.total_terminal_time += time.time() - start_time
                return -1000.0 - depth
            Connect4Minimax.total_terminal_time += time.time() - start_time
            # return get_eval(game.moves)
            return game.nnue_wrapper.accumulator_forward(game.player)
            
        # TODO: Implement check for fastest win and prune

        # TODO: Implement null move pruning
            

        start_time = time.time()
        best_value = -np.inf
        for col in range(Connect4.cols):
            if game.is_column_full(col):
                continue
            row, col = game.drop_piece(col)
            Connect4Minimax.total_non_terminal_time += time.time() - start_time
            value = -Connect4Minimax.minimax(game, (row, col), depth - 1, -beta, -alpha)
            start_time = time.time()
            game.remove_piece(row, col)
            if value > best_value:
                best_value = value
                alpha = max(alpha, best_value)
            if value >= beta:
                return best_value

        Connect4Minimax.total_non_terminal_time += time.time() - start_time
        return best_value
    
    def get_best_col(game: Connect4Game) -> int:
        best_col = None
        best_value = -np.inf
        game.evaluate_board_reset()
        for col in range(Connect4.cols):
            if game.is_column_full(col):
                continue
            row, col = game.drop_piece(col)
            value = -Connect4Minimax.minimax(game, (row, col), 2)
            print("col: ", col, "value: ", value)
            game.remove_piece(row, col)
            if value > best_value:
                best_value = value
                best_col = col
        assert best_col is not None
        return best_col