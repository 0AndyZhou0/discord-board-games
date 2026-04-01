import time

import numpy as np

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

    def minimax(game: Connect4Game, player: Color, depth: int, alpha: float = -np.inf, beta: float = np.inf) -> float:
        """Returns value of the board for the given player"""
        # Terminal Node
        winner = game.get_winner()
        if depth <= 0 or winner is not None:
            start_time = time.time()
            if winner is not None:
                Connect4Minimax.total_terminal_time += time.time() - start_time
                return -100
            Connect4Minimax.total_terminal_time += time.time() - start_time
            value = game.nnue_wrapper.accumulator_forward(player)
            if value > 0.8 or value < -0.8:
                game.print_bitboard()
                print("player: ", player)
                print("value of board: ", game.nnue_wrapper.accumulator_forward(player))
            return -game.nnue_wrapper.accumulator_forward(player)
            
        # TODO: Implement check for fastest win and prune

        # TODO: Implement null move pruning
            

        start_time = time.time()
        for col in range(Connect4.cols):
            if game.is_column_full(col):
                continue
            row, col = game.drop_piece_with_color(col, player)
            Connect4Minimax.total_non_terminal_time += time.time() - start_time
            value = -Connect4Minimax.minimax(game, -player, depth - 1, -beta, -alpha)
            start_time = time.time()
            game.remove_piece(row, col, player)
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break

        Connect4Minimax.total_non_terminal_time += time.time() - start_time
        return alpha
    
    def get_best_col(game: Connect4Game, player: Color) -> int:
        best_col = None
        best_value = -np.inf
        for col in range(Connect4.cols):
            if game.is_column_full(col):
                continue
            row, col = game.drop_piece_with_color(col, player)
            # value = -Connect4Minimax.minimax(game, -player, 0)
            value = -Connect4Minimax.minimax(game, -player, 0)
            game.remove_piece(row, col, player)
            if value > best_value:
                best_value = value
                best_col = col
        assert best_col is not None
        return best_col