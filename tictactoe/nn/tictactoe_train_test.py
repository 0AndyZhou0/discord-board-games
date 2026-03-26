import logging
from pathlib import Path
from random import shuffle

import numpy as np
import torch

from .battle import Battle
from .tictactoe import TicTacToe
from .tictactoe_mcts import TicTacToe_MCTS
from .tictactoe_nn import TicTacToeNN, TicTacToeNNWrapper

logging.basicConfig()
logger = logging.getLogger("cogs.tictactoe.nn.train_test")

class TrainTester:
    def __init__(self, nn: TicTacToeNNWrapper, c_puct: float) -> None:
        self.nn = nn
        self.c_puct = c_puct
        self.mcts = TicTacToe_MCTS(self.nn, c_puct)
        self.parent_dir = Path(__file__).parent
        self.parent_dir_model = str(self.parent_dir) + "/models"
        Path(self.parent_dir_model).mkdir(parents=True, exist_ok=True)

        self.train_sets_path = f"{self.parent_dir}/train_sets.npy"
        self.train_sets = []
        if Path(self.train_sets_path).exists():
            self.load_train_sets(self.train_sets_path)
        self.random = np.random.default_rng()

        self.train_set_max_len = 10

    def episode_from_empty(self, num_searches_per_episode_step: int = 100):  # noqa: ANN201
        tempTrainSet = []
        board = TicTacToe.get_empty_board()
        player = -1
        # Create random board weighted by number of moves
        reverse_prob = [i / sum(range(7, -1, -1)) for i in range(7, -1, -1)]
        for i in range(self.random.choice(8, p=reverse_prob)):
            action = self.random.choice(9, p=TicTacToe.get_empty_squares_mask(board)/sum(TicTacToe.get_empty_squares_mask(board)))
            next_board, next_player = TicTacToe.get_next_board(board, player, action)
            if TicTacToe.get_game_ended(next_board) is not None:
                break
            board = next_board
            player = next_player
        
        # Debug Test
        # board = np.array([[0, -1, 0], 
        #                   [-1, 1, 0], 
        #                   [1, 1, -1]])
        # player = -1

        curr_player = player
        episodeStep = 0

        # Reset mcts
        # self.mcts = TicTacToe_MCTS(self.nn, self.c_puct)

        while True:
            episodeStep += 1

            canonical_board = TicTacToe.get_canonical_board(board, curr_player)
            # probabilities = self.mcts.do_n_searches(canonical_board, num_searches_per_episode_step)
            if int(episodeStep > 5):
                probabilities = self.mcts.get_best_actions(canonical_board, num_searches_per_episode_step)
            else:
                probabilities = self.mcts.do_n_searches(canonical_board, num_searches_per_episode_step)
            # print(f"canonical_board: \n{TicTacToe.to_string(canonical_board)}\nprobabilities: {probabilities}\ncurr_player: {curr_player}")
            tempTrainSet.append((canonical_board, probabilities, curr_player))

            action = np.random.default_rng().choice(9, p=probabilities)
            next_board, next_player = TicTacToe.get_next_board(board, curr_player, action)
            board = next_board
            curr_player = next_player

            r = TicTacToe.get_game_ended(board)

            if r is not None:
                # print(r, -curr_player)
                # for ex in tempTrainSet:
                #     print(TicTacToe.to_string(board))
                #     print(TicTacToe.to_string(ex[0]))
                #     print(TicTacToe.get_canonical_board(ex[0], ex[2]))
                #     print(ex[1], ex[2], r * ex[2])
                # raise Exception("Game ended")
                return [(x[0], x[1], r * x[2]) for x in tempTrainSet]

    def save_train_sets(self, path: str = "./train_sets.npy") -> None:
        np.save(path, np.array(self.train_sets, dtype=object), allow_pickle=True)

    def load_train_sets(self, path: str = "./train_sets.npy") -> None:
        self.train_sets = np.load(path, allow_pickle=True).tolist()

    def train(self, num_iters: int = 10, num_episodes: int = 1000, num_searches_per_episode_step: int = 20, num_games_in_battle: int = 100, num_searches_per_battle: int = 10, update_threshold: float = 0.5001) -> None:
        # Battle against random
        best_against_random = 0
        random_nn = TicTacToeNNWrapper(TicTacToeNN(), self.nn.device)
        self.nn.save_model(f"{self.parent_dir_model}/temp.pt")
        curr_nn = TicTacToeNNWrapper(TicTacToeNN(), self.nn.device)
        curr_nn.load_model(f"{self.parent_dir_model}/temp.pt")

        wins_against_random, ties_against_random, _ = Battle.battles(curr_nn, random_nn, self.c_puct, num_games_in_battle, num_searches_per_battle)
        best_against_random = (wins_against_random + (0.5 * ties_against_random)) / num_games_in_battle
        logger.debug(f"Preliminary best wins against random: {best_against_random}")
        
        for iter in range(num_iters):
            if iter > 1:
                self.nn.load_model(f"{self.parent_dir_model}/best.pt")

            for episode in range(num_episodes):
                logger.debug(f"Episode {episode}")
                self.mcts = TicTacToe_MCTS(self.nn, self.c_puct)
                train_set = self.episode_from_empty(num_searches_per_episode_step)
                self.train_sets.append(train_set)

            
            if len(self.train_sets) > self.train_set_max_len:
                self.train_sets = self.train_sets[1:]
            self.save_train_sets(self.train_sets_path)


            current_train_set = []
            for train_set in self.train_sets:
                current_train_set.extend(train_set)
            shuffle(current_train_set)

            # print random samples
            for i in range(10):
                logger.debug(f"Sample {i}\n{TicTacToe.to_string(current_train_set[i][0])}")
                logger.debug(current_train_set[i][1])
                logger.debug(current_train_set[i][2])

            self.nn.save_model(f"{self.parent_dir_model}/temp.pt")
            self.new_nn = TicTacToeNNWrapper(TicTacToeNN(), self.nn.device)
            self.new_nn.load_model(f"{self.parent_dir_model}/temp.pt")


            self.new_nn.train(current_train_set, 10, 128)
            random_nn = TicTacToeNNWrapper(TicTacToeNN(), self.nn.device)

            old_wins, ties, new_wins = Battle.battles(self.nn, self.new_nn, self.c_puct, num_games_in_battle, num_searches_per_battle)
            wins_against_random, ties_against_random, _ = Battle.battles(self.new_nn, random_nn, self.c_puct, num_games_in_battle, num_searches_per_battle)

            print(f"Old Wins: {old_wins}, Ties: {ties}, New Wins: {new_wins}")
            print(f"New Wins against random: {wins_against_random}, Ties against random: {ties_against_random}")

            if ((new_wins + (0.5 * ties)) / num_games_in_battle >= update_threshold)\
                or (new_wins >= old_wins and (wins_against_random + (0.5 * ties_against_random)) / num_games_in_battle > best_against_random):
                best_against_random = (wins_against_random + (0.5 * ties_against_random)) / num_games_in_battle
                logger.debug("Updating best model")
                self.new_nn.save_model(f"{self.parent_dir_model}/best.pt")
            else:
                logger.debug("Not updating best model")
                self.nn.save_model(f"{self.parent_dir_model}/best.pt")
