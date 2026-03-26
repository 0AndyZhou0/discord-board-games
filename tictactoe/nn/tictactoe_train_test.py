import logging
from pathlib import Path
from random import shuffle

import numpy as np
import torch
from tictactoe import TicTacToe
from tictactoe_mcts import TicTacToe_MCTS
from tictactoe_nn import TicTacToeNN, TicTacToeNNWrapper

logging.basicConfig()
logger = logging.getLogger("cogs.tictactoe.nn.train_test")

class TrainTester:
    def __init__(self, nn: TicTacToeNNWrapper, c_puct: float) -> None:
        self.nn = nn
        self.c_puct = c_puct
        self.mcts = TicTacToe_MCTS(self.nn, c_puct)
        self.train_sets = []

        self.train_set_max_len = 10

    def episode_from_empty(self, num_searches_per_episode_step: int = 100):  # noqa: ANN201
        tempTrainSet = []
        board = TicTacToe.get_empty_board()
        curr_player = 1
        episodeStep = 0

        # Reset mcts
        # self.mcts = TicTacToe_MCTS(self.nn, self.c_puct)

        while True:
            episodeStep += 1

            canonical_board = TicTacToe.get_canonical_board(board, curr_player)
            # probabilities = self.mcts.do_n_searches(canonical_board, num_searches_per_episode_step)
            if int(episodeStep < 5):
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
                return [(x[0], x[1], r * x[2] * curr_player) for x in tempTrainSet]




    def train(self, num_iters: int = 10, num_episodes: int = 1000, num_searches_per_episode_step: int = 20, num_games_in_battle: int = 100, num_searches_per_battle: int = 10, update_threshold: float = 0.5001) -> None:
        for iter in range(num_iters):
            if iter > 1:
                self.nn.load_model("./best.pth")

            for episode in range(num_episodes):
                logger.debug(f"Episode {episode}")
                self.mcts = TicTacToe_MCTS(self.nn, self.c_puct)
                train_set = self.episode_from_empty(num_searches_per_episode_step)
                self.train_sets.append(train_set) # TODO: Might be wrong

            
            if len(self.train_sets) > self.train_set_max_len:
                self.train_sets = self.train_sets[1:]
            

            current_train_set = []
            for train_set in self.train_sets:
                current_train_set.extend(train_set)
            shuffle(current_train_set)
            # current_train_set = current_train_set[:len(current_train_set) // 10]

            # print random samples
            for i in range(10):
                logger.debug(len(current_train_set))
                logger.debug(TicTacToe.to_string(current_train_set[i][0]))
                logger.debug(current_train_set[i][1])
                logger.debug(current_train_set[i][2])


            self.nn.save_model("./temp.pth")
            self.new_nn = TicTacToeNNWrapper(TicTacToeNN(), self.nn.device)
            self.new_nn.load_model("./temp.pth")

            self.new_nn.train(current_train_set, 10, 64)

            old_wins, ties, new_wins = self.battles(self.nn, self.new_nn, num_games_in_battle, num_searches_per_battle)

            print(f"Old Wins: {old_wins}, Ties: {ties}, New Wins: {new_wins}")

            if (new_wins + (0.5 * ties)) / num_games_in_battle >= update_threshold:
                logger.debug("Updating best model")
                self.new_nn.save_model("./best.pth")
            else:
                logger.debug("Not updating best model")
                self.nn.save_model("./best.pth")

    # TODO: Move to separate class
    def battles(self, nn0: TicTacToeNNWrapper, nn1: TicTacToeNNWrapper, num_games: int, num_searches_per_move: int = 10, verbose: bool = False) -> tuple[int, int, int]:
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
            mcts0 = TicTacToe_MCTS(nn0, self.c_puct)
            mcts1 = TicTacToe_MCTS(nn1, self.c_puct)
            if i % 2 == 0:
                results = self.battle(mcts0, mcts1, num_searches_per_move, verbose)
                if results == 1:
                    wins_0 += 1
                    wins_x += 1
                elif results == -1:
                    wins_1 += 1
                    wins_o += 1
                else:
                    ties += 1
            else:
                results = self.battle(mcts1, mcts0, num_searches_per_move, verbose)
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

    def battle(self, mcts0: TicTacToe_MCTS, mcts1: TicTacToe_MCTS, num_searches_per_move: int = 10, verbose: bool = False) -> tuple[int, int, int]:
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
    if Path.exists("best.pth"):
        best_nn.load_model("best.pth")

    logger.setLevel(logging.DEBUG)
    logging.getLogger("cogs.tictactoe.nn.mcts").setLevel(logging.DEBUG)
    # logger.setLevel(logging.ERROR)
    tester = TrainTester(best_nn, 1)
    
    # Training
    # tester.train(num_iters=1000, num_episodes=100, num_searches_per_episode_step=20, num_searches_per_battle=10, num_games_in_battle=100, update_threshold=0.6)

    # Bot Battle
    nn0 = TicTacToeNN()
    nn1 = TicTacToeNN()
    nn0wrapper = TicTacToeNNWrapper(nn0, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    nn1wrapper = TicTacToeNNWrapper(nn1, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    nn0wrapper.load_model("best.pth")
    # nn1wrapper.load_model("best.pth")
    mcts0_score, mcts0_ties, mcts1_score = tester.battles(nn0wrapper, nn1wrapper, 100, 10, True)
    print(f"mcts0 wins: {mcts0_score}, ties: {mcts0_ties}, mcts1 wins: {mcts1_score}")
    # score = tester.battle(mcts0, mcts1, 100, True)
    # print(score)

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
