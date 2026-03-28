import logging

import numpy as np

from .connect4 import Connect4
from .connect4_nn import Connect4NNWrapper

logger = logging.getLogger("cogs.connect4.nn")


class Connect4MCTS:
    def __init__(self, nn_wrapper: Connect4NNWrapper, outputs: int = 7, c_puct: float = 1) -> None:
        self.nn_wrapper = nn_wrapper
        self.outputs = outputs
        self.c_puct = c_puct

        self.Qsa = {} # Q(s, a)
        self.Nsa = {} # Number of times action a was taken from state s
        self.Ns = {} # Number of times state s was visited
        self.Ps = {} # Policy estimated by NN

    def do_n_searches(self, canonical_board: np.array, n: int) -> np.array:
        for _ in range(n):
            self.search(canonical_board, None)
        
        hashable_board = canonical_board.tobytes()
        counts = [self.Nsa.get((hashable_board, a), 0) for a in range(self.outputs)]
        counts = [count / sum(counts) for count in counts]
        return np.array(counts)
        

    def get_best_actions(self, canonical_board: np.array, n: int) -> list[float]:
        for _ in range(n):
            self.search(canonical_board, None)
        
        hashable_board = canonical_board.tobytes()
        counts = [self.Nsa.get((hashable_board, a), 0) for a in range(self.outputs)]
        bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
        bestA = np.random.default_rng().choice(bestAs)
        probabilities = [0] * self.outputs
        probabilities[bestA] = 1
        
        return probabilities


    def search(self, canonical_board: np.array, previous_move: tuple[int, int]) -> float:
        hashable_board = canonical_board.tobytes()
        
        # Check for terminal node
        if previous_move is not None:
            winner = Connect4.get_game_win(canonical_board, *previous_move)
            if winner is not None:
                return -winner

        # Check for new leaf node
        if hashable_board not in self.Ps:
            policy, evaluation = self.nn_wrapper.evaluate_board(canonical_board)
            valid_cols = Connect4.get_valid_cols_mask(canonical_board)
            policy = policy * valid_cols
            sum_policy = np.sum(policy)
            if sum_policy > 0:
                policy = policy / sum_policy
                self.Ps[hashable_board] = policy
            else:
                logger.error("All moves are equal, policy sum is 0")
                policy = policy + valid_cols
                policy = policy / np.sum(policy)
                self.Ps[hashable_board] = policy
            
            self.Ns[hashable_board] = 0
            return -evaluation
        
        # If known node, traverse deeper
        valid_cols = Connect4.get_valid_cols(canonical_board)
        best_value = float('-inf')
        best_col = None
        for col in valid_cols:
            if (hashable_board, col) in self.Qsa:
                u = self.Qsa[(hashable_board, col)] + self.c_puct * self.Ps[hashable_board][col] * np.sqrt(self.Ns[hashable_board]) / (1 + self.Nsa[(hashable_board, col)])
            else:
                u = self.c_puct * self.Ps[hashable_board][col] * np.sqrt(self.Ns[hashable_board] + 1e-8)
            
            if u > best_value:
                best_value = u
                best_col = col
        
        next_board, move, next_player = Connect4.drop_piece_get_board(canonical_board, best_col, 1)
        next_canonical_board = Connect4.get_canonical_board(next_board, next_player)
        
        value = self.search(next_canonical_board, move)

        if (hashable_board, best_col) in self.Qsa:
            self.Qsa[(hashable_board, best_col)] = (self.Nsa[(hashable_board, best_col)] * self.Qsa[(hashable_board, best_col)] + value) / (self.Nsa[(hashable_board, best_col)] + 1)
            self.Nsa[(hashable_board, best_col)] += 1
        else:
            self.Qsa[(hashable_board, best_col)] = value
            self.Nsa[(hashable_board, best_col)] = 1

        self.Ns[hashable_board] += 1
        return -value