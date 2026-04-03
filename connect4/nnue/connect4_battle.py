import logging
import time

import numpy as np

from .connect4 import Connect4
from .connect4_minimax import Connect4Minimax
from .connect4_game import Connect4Game

logger = logging.getLogger("cogs.connect4.nn.battle")

class Battle:
    def battles(bot0: Connect4Game, bot1: Connect4Game, num_games: int = 100, verbose: bool = False) -> tuple[int, int, int]:
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
        player = -1
        for i in range(num_games):
            logger.debug(f"Game {i}")
            # Reset bots for each game
            bot0.reset()
            bot1.reset()

            if i % 2 == 0:
                results = Battle.battle(bot0, bot1, player, verbose)
                if results == 1:
                    wins_0 += 1
                    wins_red += 1
                elif results == -1:
                    wins_1 += 1
                    wins_yellow += 1
                else:
                    ties += 1
            else:
                results = Battle.battle(bot0, bot1, -player, verbose)
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

    def battle(bot0: Connect4Game, bot1: Connect4Game, player_turn: int = -1, verbose: bool = False) -> tuple[int, int, int]:
        """
        Returns:
            1 if first player wins, 0 for tie, -1 if second player wins
        """
        previous_move = (0, 0)
        curr_player = player_turn
        turn = 0

        minimax0 = Connect4Minimax()
        minimax1 = Connect4Minimax()

        total_time = 0

        while bot0.get_winner_from_move(*previous_move) is None:
            turn += 1
            if verbose:
                bot0.print_bitboard()
                logger.debug(f"Turn {turn}")
            
            start_time = time.time()
            if curr_player == -1:
                action = minimax0.iterative_deepening(bot0, 2)
            else:
                action = minimax1.iterative_deepening(bot1, 2)
            total_time += time.time() - start_time
            empty_squares = bot0.get_valid_cols_mask()
            assert len(empty_squares) > 0

            move = bot0.drop_piece(action)
            bot1.drop_piece(action)

            curr_player = -curr_player
            previous_move = move
        
        if verbose:
            bot0.print_bitboard()
            logger.debug(f"Winner: {bot0.get_winner_from_move(*previous_move)}")

        logger.debug(f"Total time: {total_time}")
        return -bot0.get_winner_from_move(*previous_move)