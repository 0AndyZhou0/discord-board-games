import logging
from pathlib import Path

import numpy as np
import torch
from deep_nn.connect4 import Connect4
from deep_nn.connect4_battle import Battle
from deep_nn.connect4_mcts import Connect4MCTS
from deep_nn.connect4_nn import Connect4NNWrapper
from deep_nn.connect4_train_test import TrainTester

logger = logging.getLogger("cogs.connect4.nn")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_nn = Connect4NNWrapper()
    random_nn = Connect4NNWrapper()
    tester = TrainTester(best_nn, random_start_board=True)

    logger.setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)

    if Path.exists(f"{tester.parent_dir_model}/best.pt"):
        best_nn.load_model(f"{tester.parent_dir_model}/best.pt")
    
    # Training
    tester.train(num_iters=1000, num_episodes=100, \
                 num_searches_per_episode_step=20, num_searches_per_battle=10, \
                    num_games_in_battle=100, update_threshold=0.55)

    # Bot Battle
    # nn0wrapper = Connect4NNWrapper()
    # nn1wrapper = Connect4NNWrapper()
    # nn0wrapper.load_model(f"{tester.parent_dir_model}/best.pt")
    # # nn1wrapper.load_model(f"{tester.parent_dir_model}/best.pt")
    # mcts0_score, mcts0_ties, mcts1_score = Battle.battles(nn0wrapper, nn1wrapper, 1, 10, 20, True)
    # print(f"mcts0 wins: {mcts0_score}, ties: {mcts0_ties}, mcts1 wins: {mcts1_score}")
    # # score = Battle.battle(mcts0, mcts1, 100, True)
    # # print(score)

    # board = Connect4.get_empty_board()
    # mcts = Connect4MCTS(best_nn)
    # player = -1
    # move = (None, None)
    # while Connect4.get_game_win(board, *move) is None:
    #     canonical_board = Connect4.get_canonical_board(board, player)
    #     probabilities = mcts.get_best_actions(canonical_board, 100)
    #     action = np.argmax(probabilities)
    #     board, move, player = Connect4.drop_piece_get_board(board, action, player)
    #     Connect4.display_board(board)