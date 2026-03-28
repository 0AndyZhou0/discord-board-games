import logging
from pathlib import Path

import numpy as np
import torch
from nn.battle import Battle
from nn.tictactoe import TicTacToe
from nn.tictactoe_mcts import TicTacToe_MCTS
from nn.tictactoe_nn import (
    TicTacToeNN,
    TicTacToeNNWrapper,
)
from nn.tictactoe_train_test import TrainTester

logger = logging.getLogger("cogs.tictactoe.nn.train_test")

def evaluate_board(nn: TicTacToeNNWrapper, board: np.array) -> tuple[torch.Tensor, torch.Tensor]:
    mcts = TicTacToe_MCTS(nn, 1)
    print(TicTacToe.to_string(board))
    canonical_board = TicTacToe.get_canonical_board(board, TicTacToe.get_current_player(board))
    print(TicTacToe.to_string(canonical_board))
    print("nn:", mcts.do_n_searches(canonical_board, 10))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_nn = TicTacToeNNWrapper(TicTacToeNN(), torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    random_nn = TicTacToeNNWrapper(TicTacToeNN(), torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    tester = TrainTester(best_nn, 1)

    if Path.exists(f"{tester.parent_dir_model}/best.pt"):
        best_nn.load_model(f"{tester.parent_dir_model}/best.pt")

    logger.setLevel(logging.DEBUG)
    logging.getLogger("cogs.tictactoe.nn.mcts").setLevel(logging.DEBUG)
    # logger.setLevel(logging.ERROR)
    
    # Training
    tester.train(num_iters=1000, num_episodes=100, \
                 num_searches_per_episode_step=20, num_searches_per_battle=10, \
                    num_games_in_battle=100, update_threshold=0.5\
                    , random_test=True)

    # Bot Battle
    # nn0 = TicTacToeNN()
    # nn1 = TicTacToeNN()
    # nn0wrapper = TicTacToeNNWrapper(nn0, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # nn1wrapper = TicTacToeNNWrapper(nn1, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # nn0wrapper.load_model(f"{tester.parent_dir_model}/best.pt")
    # # nn1wrapper.load_model(f"{tester.parent_dir_model}/best.pt")
    # mcts0_score, mcts0_ties, mcts1_score = Battle.battles(nn0wrapper, nn1wrapper, 1, 1000, 10, False)
    # print(f"mcts0 wins: {mcts0_score}, ties: {mcts0_ties}, mcts1 wins: {mcts1_score}")
    # # score = Battle.battle(mcts0, mcts1, 100, True)
    # # print(score)

    # Manual Test
    # board = TicTacToe.get_empty_board()
    # print("Best NN:")
    # evaluate_board(best_nn, board)
    # print("Random NN:")
    # evaluate_board(random_nn, board)

    # board = np.array([[-1, 0, 0], 
    #                   [0, 1, 0], 
    #                   [1, 0, -1]])
    # print("Best NN:")
    # evaluate_board(best_nn, board)
    # print("Random NN:")
    # evaluate_board(random_nn, board)

    # board = np.array([[-1, 0, -1], 
    #                   [0, 1, 0], 
    #                   [1, 0, -1]])
    # print("Best NN:")
    # evaluate_board(best_nn, board)
    # print("Random NN:")
    # evaluate_board(random_nn, board)

    # board = np.array([[-1, 1, -1], 
    #                   [0, 1, 0], 
    #                   [1, 0, -1]])
    # print("Best NN:")
    # evaluate_board(best_nn, board)
    # print("Random NN:")
    # evaluate_board(random_nn, board)

    # board = np.array([[1, -1, 0], 
    #                   [0, -1, 0], 
    #                   [1, 0, 0]])
    # print("Best NN:")
    # evaluate_board(best_nn, board)
    # print("Random NN:")
    # evaluate_board(random_nn, board)
