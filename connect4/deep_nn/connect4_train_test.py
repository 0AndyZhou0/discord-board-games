import logging
from pathlib import Path
from random import shuffle

import numpy as np

from .connect4 import Connect4
from .connect4_battle import Battle
from .connect4_mcts import Connect4MCTS
from .connect4_nn import Connect4NNWrapper

logging.basicConfig()
logger = logging.getLogger("cogs.connect4.nn")

class TrainTester:
    def __init__(self, nn: Connect4NNWrapper, outputs: int = 7, c_puct: float = 1) -> None:
        self.nn = nn
        self.outputs = outputs
        self.c_puct = c_puct
        self.mcts = Connect4MCTS(self.nn, outputs, c_puct)
        self.parent_dir = Path(__file__).parent
        self.parent_dir_model = str(self.parent_dir) + "/models"
        Path(self.parent_dir_model).mkdir(parents=True, exist_ok=True)

        self.train_sets_path = f"{self.parent_dir}/train_sets.npy"
        self.train_sets = []
        if Path(self.train_sets_path).exists():
            self.load_train_sets(self.train_sets_path)
        self.random = np.random.default_rng()

        self.train_set_max_len = 10

    def episode(self, num_searches_per_episode_step: int = 100):  # noqa: ANN201
        tempTrainSet = []
        board = Connect4.get_empty_board()
        player = -1

        # TODO: Create random board weighted by number of moves
        board, player = Connect4.get_random_board(10)

        curr_player = player
        episodeStep = 0

        while True:
            episodeStep += 1

            canonical_board = Connect4.get_canonical_board(board, curr_player)
            if int(episodeStep > 10):
                probabilities = self.mcts.get_best_actions(canonical_board, num_searches_per_episode_step)
            else:
                probabilities = self.mcts.do_n_searches(canonical_board, num_searches_per_episode_step)
            tempTrainSet.append((canonical_board, probabilities, curr_player))

            action = np.random.default_rng().choice(7, p=probabilities)
            next_board, move, next_player = Connect4.drop_piece_get_board(board, action, curr_player)
            board = next_board
            curr_player = next_player

            r = Connect4.get_game_win(board, *move)

            if r is not None:
                return [(x[0], x[1], r * x[2]) for x in tempTrainSet]

    def save_train_sets(self, path: str = "./train_sets.npy") -> None:
        np.save(path, np.array(self.train_sets, dtype=object), allow_pickle=True)

    def load_train_sets(self, path: str = "./train_sets.npy") -> None:
        self.train_sets = np.load(path, allow_pickle=True).tolist()

    def train(self, num_iters: int = 1000, num_episodes: int = 100, num_searches_per_episode_step: int = 20, num_games_in_battle: int = 100, num_searches_per_battle: int = 20, update_threshold: float = 0.55) -> None:
        for iter in range(num_iters):
            if iter > 1:
                self.nn.load_model(f"{self.parent_dir_model}/best.pt")

            session_train_set = []
            for episode in range(num_episodes):
                logger.debug(f"Episode {episode}")
                self.mcts = Connect4MCTS(self.nn, self.outputs, self.c_puct)
                train_set = self.episode(num_searches_per_episode_step)
                session_train_set.extend(train_set)

            self.train_sets.append(session_train_set)
            
            if len(self.train_sets) > self.train_set_max_len:
                self.train_sets = self.train_sets[1:]
            self.save_train_sets(self.train_sets_path)


            current_train_set = []
            for train_set in self.train_sets:
                current_train_set.extend(train_set)
            shuffle(current_train_set)

            # Debug print samples
            # for i in range(10):
            #     logger.debug(f"Sample {i}: \n{current_train_set[i]}")

            self.nn.save_model(f"{self.parent_dir_model}/temp.pt")
            new_nn = Connect4NNWrapper()
            new_nn.load_model(f"{self.parent_dir_model}/temp.pt")


            new_nn.train(current_train_set, 10, 64)
            old_wins, ties, new_wins = Battle.battles(self.nn, new_nn, self.c_puct, num_games_in_battle, num_searches_per_battle)
            print(f"Old Wins: {old_wins}, Ties: {ties}, New Wins: {new_wins}")

            # if ((new_wins + (0.5 * ties)) / num_games_in_battle >= update_threshold):
            if ((new_wins + (0.5 * ties)) / num_games_in_battle >= update_threshold):
                logger.debug("Updating best model")
                new_nn.save_model(f"{self.parent_dir_model}/best.pt")
            else:
                logger.debug("Not updating best model")
                self.nn.save_model(f"{self.parent_dir_model}/best.pt")
