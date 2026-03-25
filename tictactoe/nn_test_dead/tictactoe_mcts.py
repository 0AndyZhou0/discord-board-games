import logging
from copy import deepcopy

import numpy as np
import torch
from tictactoe_nn import TicTacToeNN, TicTacToeNNWrapper

from tictactoe import TicTacToe

logger = logging.getLogger("cogs.tictactoe.nn.mcts")


class TicTacToe_MCTS:
    def __init__(self, nn: TicTacToeNNWrapper, c_puct: float) -> None:
        self.nn = nn

        self.Qsa = {} # Q(s, a)
        self.Nsa = {} # Number of times action a was taken from state s
        self.Ns = {} # Number of times state s was visited
        self.Ps = {} # Policy estimated by NN
        self.c_puct = c_puct

        # self.Ts = {} # Terminal Nodes

    def do_n_searches(self, game: TicTacToe, n: int) -> np.array:
        for _ in range(n):
            self.search(game)
        
        hashable_board = str(game)
        counts = [self.Nsa.get((hashable_board, a), 0) for a in range(9)]
        counts = [count / sum(counts) for count in counts]
        return np.array(counts)
        

    def get_best_actions(self, game: TicTacToe, n: int) -> list[float]:
        for _ in range(n):
            self.search(game)
        
        hashable_board = str(game)
        # for a in range(9):
        #     print(self.Nsa.get((hashable_board, a), 0))
        counts = [self.Nsa.get((hashable_board, a), 0) for a in range(9)]
        bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
        bestA = np.random.default_rng().choice(bestAs)
        probabilities = [0] * 9
        probabilities[bestA] = 1
        
        return probabilities


    def search(self, game: TicTacToe) -> float:
        curr_game = TicTacToe()
        curr_game.board = deepcopy(game.board)
        hashable_board = str(curr_game)
        # Return if terminal node
        winner = curr_game.is_game_ended() # -1 = X win, 0 = tie, 1 = O win
        if winner is not None:
            # print("terminal node")
            # self.Ns[hashable_board] = 0
            return -winner if game.get_current_player() == 1 else winner
        
        if hashable_board not in self.Ps:
            # New leaf Node
            # print("new leaf node")
            policy, evaluation = self.nn.evaluate_board(curr_game.board)
            empty_squares = curr_game.get_empty_squares_mask()
            policy = policy * empty_squares
            sum_policy = policy.sum()
            if sum_policy > 0:
                policy = policy / sum_policy
                self.Ps[hashable_board] = policy
            else:
                logger.error("policy sum is 0")
                policy = policy + empty_squares
                policy = policy / policy.sum()
                self.Ps[hashable_board] = policy
            
            self.Ns[hashable_board] = 0
            return -evaluation
        
        # Choose move
        # print("choosing move")
        empty_squares = curr_game.get_empty_squares_mask()
        best_value = -float('inf')
        best_action = -1

        for a in range(9):
            if empty_squares[a]:
                if (hashable_board, a) in self.Qsa:
                    u = self.Qsa[(hashable_board, a)] + self.c_puct * self.Ps[hashable_board][a] * np.sqrt(self.Ns[hashable_board]) / (1 + self.Nsa[(hashable_board, a)])
                else:
                    u = self.c_puct * self.Ps[hashable_board][a] * np.sqrt(self.Ns[hashable_board] + 1e-8)

                if u > best_value:
                    best_value = u
                    best_action = a

        # Play move
        curr_game.play_move((best_action // 3, best_action % 3), curr_game.get_current_player())
        
        value = self.search(curr_game)
        
        if (hashable_board, best_action) in self.Qsa:
            self.Qsa[(hashable_board, best_action)] = (self.Nsa[(hashable_board, best_action)] * self.Qsa[(hashable_board, best_action)] + value) / (self.Nsa[(hashable_board, best_action)] + 1)
            self.Nsa[(hashable_board, best_action)] += 1
        else:
            self.Qsa[(hashable_board, best_action)] = value
            self.Nsa[(hashable_board, best_action)] = 1

        self.Ns[hashable_board] += 1
        return -value
    
if __name__ == "__main__":
    game = TicTacToe()
    nnwrapper = TicTacToeNNWrapper(TicTacToeNN(), torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    mcts = TicTacToe_MCTS(nnwrapper, 1)
    prob_move_1 = mcts.do_n_searches(game, 100)
    print(prob_move_1)