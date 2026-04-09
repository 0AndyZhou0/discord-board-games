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
        self.bounds_table = Connect4TranspositionTable()
        self.values_table = Connect4TranspositionTable()
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
                return -10000.0 - depth
            self.total_terminal_time += time.time() - start_time
            # return get_eval(game.moves)
            return game.nnue_wrapper.accumulator_forward(game.player)
            
        # Prune based on number of moves left
        min_value = -(Connect4.cols * Connect4.rows - 2 - len(game.moves)) // 2
        if alpha < min_value:
            alpha = min_value
            if alpha >= beta:
                return alpha
        
        max_value = (Connect4.cols * Connect4.rows - 1 - len(game.moves)) // 2
        if beta > max_value:
            beta = max_value
            if alpha >= beta:
                return beta

        # Prune with table bounds
        if self.bounds_table.contains(game.moves):
            table_value = self.bounds_table.get(game.moves)
            if table_value > 2 * 18 + 1:
                min_value = table_value - 3 * 18 - 2
                if alpha < min_value:
                    alpha = min_value
                    if alpha >= beta:
                        return alpha
            else:
                max_value = table_value - 18 - 1
                if beta > max_value:
                    beta = max_value
                    if alpha >= beta:
                        return beta



        # Sort columns by evaluation
        evals = [self.values_table.get(game.moves + str(col)) for col in range(Connect4.cols)]
        order = np.flip(np.argsort(evals))


        start_time = time.time()
        for col in order:
            if game.is_column_full(col) or evals[col] < -500:
                continue
            row, col = game.drop_piece(col)
            self.total_non_terminal_time += time.time() - start_time
            value = -self.minimax(game, (row, col), depth - 1, -beta, -alpha)
            start_time = time.time()
            game.remove_piece(row, col)
            if value > alpha:
                alpha = value
            if value >= beta:
                self.bounds_table.add(game.moves, value + 3 * 18 + 2)
                return value

        self.bounds_table.add(game.moves, alpha + 18 + 1)
        self.values_table.add(game.moves, alpha)
        self.total_non_terminal_time += time.time() - start_time
        return alpha
    
    def iterative_deepening(self, game: Connect4Game, depth: int = 5) -> int:
        for d in range(1, depth + 1):
            col = self.get_best_col(game, d)
        return col
    
    def get_best_col(self, game: Connect4Game, depth: int) -> int:
        best_cols = []
        best_value = -np.inf
        # Get column order
        evals = [self.values_table.get(game.moves + str(col)) for col in range(Connect4.cols)]
        order = np.flip(np.argsort(evals))
        for col in order:
            if game.is_column_full(col) or evals[col] < -500:
                continue
            row, col = game.drop_piece(col)
            # value = -self.minimax(game, (row, col), depth-1, 0, 1)
            value = -self.minimax(game, (row, col), depth-1, -18, 18)
            game.remove_piece(row, col)
            if value > 500: # Instantly play winning move
                return col
            if value > best_value:
                best_value = value
                best_cols = [col]
            elif value == best_value:
                best_cols.append(col)
        self.values_table.add(game.moves, best_value)
        return np.random.choice(best_cols)