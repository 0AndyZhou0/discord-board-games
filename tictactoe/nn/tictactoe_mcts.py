import logging

import numpy as np

from .tictactoe import TicTacToe
from .tictactoe_nn import TicTacToeNNWrapper

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

    def do_n_searches(self, canonical_board: np.array, n: int) -> np.array:
        for _ in range(n):
            self.search(canonical_board)
        
        hashable_board = TicTacToe.to_string(canonical_board)
        counts = [self.Nsa.get((hashable_board, a), 0) for a in range(9)]
        counts = [count / sum(counts) for count in counts]
        return np.array(counts)
        

    def get_best_actions(self, canonical_board: np.array, n: int) -> list[float]:
        for _ in range(n):
            self.search(canonical_board)
        
        hashable_board = TicTacToe.to_string(canonical_board)
        counts = [self.Nsa.get((hashable_board, a), 0) for a in range(9)]
        bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
        bestA = np.random.default_rng().choice(bestAs)
        probabilities = [0] * 9
        probabilities[bestA] = 1
        
        return probabilities


    def search(self, canonical_board: np.array) -> float:
        hashable_board = TicTacToe.to_string(canonical_board)
        # Return if terminal node
        winner = TicTacToe.get_game_ended(canonical_board) # 1 = Player wins, 0 = tie, -1 = Player loses
        if winner is not None:
            # if len(TicTacToe.get_empty_squares(canonical_board)) % 2 == 1:
            #     print(winner * (-1 ** (len(TicTacToe.get_empty_squares(canonical_board)) % 2)))
            #     print(f"winner: {winner}, canonical_board: \n{TicTacToe.to_string(canonical_board)}")
            # print("terminal node")
            # self.Ns[hashable_board] = 0
            return -winner
        
        if hashable_board not in self.Ps:
            # New leaf Node
            # print("new leaf node")
            policy, evaluation = self.nn.evaluate_board(canonical_board)
            empty_squares = TicTacToe.get_empty_squares_mask(canonical_board)
            policy = policy * empty_squares
            sum_policy = np.sum(policy)
            if sum_policy > 0:
                policy = policy / sum_policy
                self.Ps[hashable_board] = policy
            else:
                logger.error("policy sum is 0")
                policy = policy + empty_squares
                policy = policy / np.sum(policy)
                self.Ps[hashable_board] = policy
            
            self.Ns[hashable_board] = 0
            return evaluation
        
        # Choose move
        # print("choosing move")
        empty_squares = TicTacToe.get_empty_squares_mask(canonical_board)
        best_value = -float('inf')
        best_action = -1

        for a in range(9):
            if empty_squares[a] != 0:
                if (hashable_board, a) in self.Qsa:
                    u = self.Qsa[(hashable_board, a)] + self.c_puct * self.Ps[hashable_board][a] * np.sqrt(self.Ns[hashable_board]) / (1 + self.Nsa[(hashable_board, a)])
                else:
                    u = self.c_puct * self.Ps[hashable_board][a] * np.sqrt(self.Ns[hashable_board] + 1e-8)

                if u > best_value:
                    best_value = u
                    best_action = a

        # Play move
        next_board, next_player = TicTacToe.get_next_board(canonical_board, 1, best_action)
        next_canonical_board = TicTacToe.get_canonical_board(next_board, next_player)
        
        value = self.search(next_canonical_board)
        # if value > 0.5 and value < 1 or value < -0.5 and value > -1:
        #     # logger.debug(f"next_player: {next_player}")
        #     logger.debug(f"canonical_board: \n{TicTacToe.to_string(canonical_board)}")
        #     for a in range(9):
        #         if (hashable_board, a) in self.Qsa:
        #             logger.debug(f"Qsa[{a}]: {self.Qsa[(hashable_board, a)]}")
        #     logger.debug(f"next_canonical_board (value: {value}): \n{TicTacToe.to_string(next_canonical_board)}")
        
        if (hashable_board, best_action) in self.Qsa:
            self.Qsa[(hashable_board, best_action)] = (self.Nsa[(hashable_board, best_action)] * self.Qsa[(hashable_board, best_action)] + value) / (self.Nsa[(hashable_board, best_action)] + 1)
            self.Nsa[(hashable_board, best_action)] += 1
        else:
            self.Qsa[(hashable_board, best_action)] = value
            self.Nsa[(hashable_board, best_action)] = 1

        self.Ns[hashable_board] += 1
        return -value
    