import logging

import numpy as np

from .connect4 import Connect4
from .connect4_mcts import Connect4MCTS
from .connect4_nn import Connect4NNWrapper

logger = logging.getLogger("cogs.connect4.nn.battle")

class Battle:
    def battles(nn0: Connect4NNWrapper, nn1: Connect4NNWrapper, c_puct: float = 1, num_games: int = 100, num_searches_per_move: int = 20, verbose: bool = False) -> tuple[int, int, int]:
        """
        Returns:
            (nn0 wins, ties, nn1 wins)
        """
        assert num_games % 2 == 0
        wins_0 = 0
        ties = 0
        wins_1 = 0
        wins_red = 0
        wins_yellow = 0
        for i in range(num_games):
            # Reset mcts for each game
            mcts0 = Connect4MCTS(nn0)
            mcts1 = Connect4MCTS(nn1)
            if i % 2 == 0:
                # Create random board 
                board, player = Connect4.get_random_board(10) 
                results = Battle.battle(mcts0, mcts1, board, num_searches_per_move, verbose)
                if results == 1:
                    wins_0 += 1
                    wins_red += 1
                elif results == -1:
                    wins_1 += 1
                    wins_yellow += 1
                else:
                    ties += 1
            else:
                results = Battle.battle(mcts1, mcts0, board, num_searches_per_move, verbose)
                if results == 1:
                    wins_1 += 1
                    wins_red += 1
                elif results == -1:
                    wins_0 += 1
                    wins_yellow += 1
                else:
                    ties += 1
        
        logger.debug(f"red wins: {wins_red}, yellow wins: {wins_yellow}, ties: {ties}")
        return wins_0, ties, wins_1

    def battle(mcts0: Connect4MCTS, mcts1: Connect4MCTS, board: np.array = None, num_searches_per_move: int = 20, verbose: bool = False) -> tuple[int, int, int]:
        """
        Returns:
            1 if first player wins, 0 for tie, -1 if second player wins
        """
        if board is None:
            board = Connect4.get_empty_board()
        previous_move = (None, None)
        curr_player = -1
        turn = 0

        while Connect4.get_game_win(board, *previous_move) is None:
            turn += 1
            if verbose:
                Connect4.display_board(board)
                logger.debug(f"Turn {turn}")
            
            canonical_board = Connect4.get_canonical_board(board, curr_player)
            if turn % 2 == 1:
                action = np.argmax(mcts0.get_best_actions(canonical_board, num_searches_per_move))
            else:
                action = np.argmax(mcts1.get_best_actions(canonical_board, num_searches_per_move))
            empty_squares = Connect4.get_valid_cols_mask(board)
            assert len(empty_squares) > 0

            next_board, move, next_player = Connect4.drop_piece_get_board(board, action, curr_player)
            board, previous_move, curr_player = next_board, move, next_player
        
        if verbose:
            Connect4.display_board(board)
            logger.debug(f"Winner: {Connect4.get_game_win(board, *previous_move)}")

        return -Connect4.get_game_win(board, *previous_move)