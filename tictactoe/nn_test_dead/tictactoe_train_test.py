import logging
from copy import deepcopy
from pathlib import Path
from random import shuffle

import numpy as np
import torch
from tictactoe_mcts import TicTacToe_MCTS
from tictactoe_nn import TicTacToeNN, TicTacToeNNWrapper

from tictactoe import TicTacToe

logging.basicConfig()
logger = logging.getLogger("cogs.tictactoe.nn.train_test")

class TrainTester:
    def __init__(self, nn: TicTacToeNNWrapper, c_puct: float) -> None:
        self.nn = nn
        self.c_puct = c_puct
        self.mcts = TicTacToe_MCTS(self.nn, c_puct)
        self.train_sets = []

        self.train_set_max_len = 1000

    def episode(self, game: TicTacToe, num_searches_per_episode: int = 100) -> list[tuple[list[list[int]], list[float], float]]:
        tempTrainSet = []
        episodeStep = 0

        # Reset mcts
        # self.mcts = TicTacToe_MCTS(self.nn, self.c_puct)

        while True:
            episodeStep += 1

            probabilities = self.mcts.do_n_searches(game, num_searches_per_episode)
            tempTrainSet.append((game.board, probabilities, game.get_current_player()))
            # evaluation = self.mcts.search(game)
#             logger.debug(f"""Step: {episodeStep}, Evaluation: {evaluation}
# Probabilities: 
# {probabilities[0:3]}
# {probabilities[3:6]}
# {probabilities[6:9]}
# {game}""")

            move = np.random.default_rng().choice(9, p=probabilities)
            game.play_move((move // 3, move % 3), game.get_current_player())

            if game.is_game_ended() != None:
                return [(x[0], x[1], game.is_game_ended() * game.get_current_player()) for x in tempTrainSet]
                # return [(x[0], x[1], (-game.is_game_ended() if x[2] == game.get_current_player() else game.is_game_ended())) for x in tempTrainSet]



    def train(self, game: TicTacToe, num_iters: int = 10, num_episodes: int = 1000, num_searches_per_episode: int = 100, num_games_in_battle: int = 100, num_searches_per_battle: int = 10, update_threshold: float = 0.55) -> None:
        for iter in range(num_iters):
            if iter > 1:
                self.nn.load_model("./best.pth")

            for episode in range(num_episodes):
                curr_game = deepcopy(game)
                self.mcts = TicTacToe_MCTS(self.nn, self.c_puct)
                train_set = self.episode(curr_game, num_searches_per_episode)
                self.train_sets.append(train_set)
        
            
            if len(self.train_sets) > self.train_set_max_len:
                self.train_sets = self.train_sets[1:]
            

            current_train_set = []
            for train_set in self.train_sets:
                current_train_set.extend(train_set)
            shuffle(current_train_set)
            current_train_set = current_train_set[:len(current_train_set) // 10]


            self.nn.save_model("./temp.pth")
            self.new_nn = TicTacToeNNWrapper(TicTacToeNN(), self.nn.device)
            self.new_nn.load_model("./temp.pth")

            self.new_nn.train(current_train_set, 10, 100)

            old_mcts = TicTacToe_MCTS(self.nn, self.c_puct)
            new_mcts = TicTacToe_MCTS(self.new_nn, self.c_puct)

            old_wins, ties, new_wins = self.battle(game, old_mcts, new_mcts, num_games_in_battle, num_searches_per_battle)

            print(f"Old Wins: {old_wins}, Ties: {ties}, New Wins: {new_wins}")

            if new_wins / (old_wins + new_wins) > update_threshold:
                self.new_nn.save_model("./best.pth")
            else:
                self.nn.save_model("./best.pth")




    def battle(self, game: TicTacToe, mcts0: TicTacToe_MCTS, mcts1: TicTacToe_MCTS, num_games: int, num_searches_per_move: int = 100, verbose: bool = False) -> tuple[int, int, int]:
        """
        Returns:
            (nn0 wins, ties, nn1 wins)
        """
        starting_player = 0 # 0 = nn0, 1 = nn1
        wins_0 = 0
        ties = 0
        wins_1 = 0
        wins_x = 0
        wins_o = 0
        for _ in range(num_games):
            curr_player = starting_player
            curr_game = TicTacToe()
            curr_game.board = deepcopy(game.board)
            turn = 0

            while curr_game.is_game_ended() is None:
                if curr_player == 0:
                    probabilities = mcts0.get_best_actions(curr_game, num_searches_per_move)
                    move = np.argmax(probabilities)
                else:
                    probabilities = mcts1.get_best_actions(curr_game, num_searches_per_move)
                    move = np.argmax(probabilities)
                if verbose:
                    print(mcts0.do_n_searches(curr_game, num_searches_per_move))
                    print(mcts1.do_n_searches(curr_game, num_searches_per_move))
                    print(curr_game)
                
                empty_squares = curr_game.get_empty_squares_mask()
                if empty_squares[move] == 0:
                    raise Exception("Invalid move")
                

                curr_game.play_move((move // 3, move % 3), curr_game.get_current_player())
                curr_player = 1 - curr_player
                turn += 1

            match curr_game.is_game_ended():
                case -1:
                    wins_x += 1
                    if starting_player == 0:
                        wins_0 += 1
                    else:
                        wins_1 += 1
                case 1e-4:
                    ties += 1
                case 1:
                    wins_o += 1
                    if starting_player == 0:
                        wins_1 += 1
                    else:
                        wins_0 += 1
                case _:
                    logger.error("Invalid game state. Game ended: ", curr_game.is_game_ended())
                    logger.error("Game: \n", curr_game)

            starting_player = 1 - starting_player

        logger.debug(f"x wins: {wins_x}, o wins: {wins_o}, ties: {ties}")
        return wins_0, ties, wins_1

def battle_test(mcts0: TicTacToe_MCTS, mcts1: TicTacToe_MCTS, matches: int, rounds: int, searches_per_move: int = 100, verbose: bool = False) -> None:
    tester = TrainTester(mcts0.nn, 1)
    for _ in range(matches):
        wins_0, ties, wins_1 = tester.battle(TicTacToe(), mcts0, mcts1, rounds, searches_per_move, verbose=verbose)
        print(f"nn0 wins: {wins_0}, ties: {ties}, nn1 wins: {wins_1}")

def manual_test(mcts: TicTacToe_MCTS, game: TicTacToe, num_searches_per_move: int = 100) -> None:
    player_turn = input("0 or 1: ")
    if player_turn == "0":
        move = input("Move(x, y)->(3*x + y): ")
        game.play_move((int(move) // 3, int(move) % 3), game.get_current_player())
    turn = "bot"
    while game.is_game_ended() is None:
        if turn == "bot":
            probabilities = mcts.get_best_actions(game, num_searches_per_move)
            move = np.argmax(probabilities)
            game.play_move((move // 3, move % 3), game.get_current_player())
            print(game)
        elif turn == "human":
            move = input("Move(x, y)->(3*x + y): ")
            game.play_move((int(move) // 3, int(move) % 3), game.get_current_player())
            print(game)
        turn = "bot" if turn == "human" else "human"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_nn = TicTacToeNNWrapper(TicTacToeNN(), torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    random_nn = TicTacToeNNWrapper(TicTacToeNN(), torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if Path.exists("best.pth"):
        best_nn.load_model("best.pth")

    logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.ERROR)
    
    # Training
    tester = TrainTester(best_nn, 1)
    game = TicTacToe()
    tester.train(game, num_iters=10, num_episodes=1000, num_searches_per_episode=100, num_searches_per_battle=100, num_games_in_battle=50, update_threshold=0.55)

    # Bot Battle
    # nn0 = TicTacToeNN()
    # nn1 = TicTacToeNN()
    # nn0wrapper = TicTacToeNNWrapper(nn0, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # nn1wrapper = TicTacToeNNWrapper(nn1, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # nn0wrapper.load_model("best.pth")
    # # nn1wrapper.load_model("best.pth")
    # mcts0 = TicTacToe_MCTS(nn0wrapper, 1)
    # mcts1 = TicTacToe_MCTS(nn1wrapper, 1)
    # # battle_test(mcts0, mcts1, 1, 100, 100, False)
    # # battle_test(mcts0, mcts1, 1, 100, 100, False)
    # battle_test(mcts0, mcts1, 1, 2, 100, True)

    # Manual Test
    # mcts = TicTacToe_MCTS(random_nn, 1)
    # mcts = TicTacToe_MCTS(best_nn, 1)
    # game = TicTacToe()
    # game.board = [[-1, -1, 1], [0, 1, 0], [0, 1, -1]]
    # print(game)
    # print(mcts.do_n_searches(game, 1000))
    # print(mcts.nn.evaluate_board(game.board))
    # print("-" * 100)
    # game.board = [[-1, -1, 1], [-1, 1, 0], [0, 1, -1]]
    # print(game)
    # print(mcts.do_n_searches(game, 1000))
    # print(mcts.nn.evaluate_board(game.board))

    # manual_test(mcts, game, 1000)