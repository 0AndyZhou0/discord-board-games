from pathlib import Path

import numpy as np

from .connect4_minimax import Connect4Minimax

from .connect4_color import Color

from .connect4_game import Connect4Game


class Connect4Translator:
    def __init__(self) -> None:
        self.game = Connect4Game()
        self.path_dir = Path(__file__).parent
        model_path = self.path_dir / "models" / "best.pt"
        self.game.load_model(model_path)
        self.minimax = Connect4Minimax()

    # def get_best_col_from_board(self, board: np.array, player: Color) -> int:
    #     game = self.game
    #     red_bitboard, yellow_bitboard = game.to_bitboards(board)
    #     game.red_bitboard = red_bitboard
    #     game.yellow_bitboard = yellow_bitboard
    #     game.player = player
    #     best_col = Connect4Minimax.get_best_col(game, player)
    #     return best_col
    
    def get_best_col_from_board(self, board: np.array, player: Color, moves: str) -> int:
        game = self.game
        red_bitboard, yellow_bitboard = game.to_bitboards(board)
        game.red_bitboard = red_bitboard
        game.yellow_bitboard = yellow_bitboard
        game.player = player
        game.moves = moves
        game.evaluate_board_reset()
        depth = min(4 + (len(moves) // 8), 42 - len(moves) - 1)
        best_col = self.minimax.get_best_col(game, depth)
        return best_col