import logging

import numpy as np

from .tictactoe import TicTacToe
from .tictactoe_mcts import TicTacToe_MCTS
from .tictactoe_nn import TicTacToeNNWrapper

logger = logging.getLogger("cogs.tictactoe.nn.battle")

class Battle:
    def battles(nn0: TicTacToeNNWrapper, nn1: TicTacToeNNWrapper, c_puct: float, num_games: int, num_searches_per_move: int = 10, verbose: bool = False) -> tuple[int, int, int]:
        """
        Returns:
            (nn0 wins, ties, nn1 wins)
        """
        wins_0 = 0
        ties = 0
        wins_1 = 0
        wins_x = 0
        wins_o = 0
        for i in range(num_games):
            # Reset mcts for each game
            mcts0 = TicTacToe_MCTS(nn0, c_puct)
            mcts1 = TicTacToe_MCTS(nn1, c_puct)
            if i % 2 == 0:
                results = Battle.battle(mcts0, mcts1, num_searches_per_move, verbose)
                if results == 1:
                    wins_0 += 1
                    wins_x += 1
                elif results == -1:
                    wins_1 += 1
                    wins_o += 1
                else:
                    ties += 1
            else:
                results = Battle.battle(mcts1, mcts0, num_searches_per_move, verbose)
                if results == 1:
                    wins_1 += 1
                    wins_x += 1
                elif results == -1:
                    wins_0 += 1
                    wins_o += 1
                else:
                    ties += 1
        
        logger.debug(f"x wins: {wins_x}, o wins: {wins_o}, ties: {ties}")
        return wins_0, ties, wins_1

    def battle(mcts0: TicTacToe_MCTS, mcts1: TicTacToe_MCTS, num_searches_per_move: int = 10, verbose: bool = False) -> tuple[int, int, int]:
        """
        Returns:
            1 if first player wins, 0 for tie, -1 if second player wins
        """
        board = TicTacToe.get_empty_board()
        curr_player = -1
        turn = 0

        while TicTacToe.get_game_ended(board) is None:
            turn += 1
            if verbose:
                logger.debug(f"Turn {turn}\n{TicTacToe.to_string(board)}")
            
            canonical_board = TicTacToe.get_canonical_board(board, curr_player)
            if turn % 2 == 1:
                action = np.argmax(mcts0.get_best_actions(canonical_board, num_searches_per_move))
            else:
                action = np.argmax(mcts1.get_best_actions(canonical_board, num_searches_per_move))
            empty_squares = TicTacToe.get_empty_squares_mask(board)
            assert len(empty_squares) > 0

            next_board, next_player = TicTacToe.get_next_board(board, curr_player, action)
            board, curr_player = next_board, next_player
        
        if verbose:
            print(f"Final board:\n{TicTacToe.to_string(board)}")
            print(f"Winner: {TicTacToe.get_game_ended(board)}")
        return -TicTacToe.get_game_ended(board)