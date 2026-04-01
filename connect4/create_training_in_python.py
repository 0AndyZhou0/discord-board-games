from pathlib import Path
import time

import numpy as np

from nnue.connect4_game import Connect4Game

from nnue.external_functions import get_move_evals, get_eval


ROWS = 6
COLUMNS = 7
path = Path(__file__).parent

def game_from_moves(moves: bytes) -> Connect4Game:
    moves = moves.decode()
    game = Connect4Game()
    for move in moves:
        col = int(move) - 1
        print(col)
        game.drop_piece(col)
    return game

def do_episode() -> list[tuple[np.longlong, np.longlong, np.bool, np.long]]: # red_bitboard, yellow_bitboard, player, reward
    game = Connect4Game()
    training_data = []
    winner = None
    while winner is None:
        # Store game state
        eval = get_eval(game.moves) # positive if current player wins
        training_data.append((game.red_bitboard, game.yellow_bitboard, game.player, eval))

        # More likely to choose a better move
        valid_cols = game.get_valid_cols_mask()
        move_evals = np.array(get_move_evals(game.moves))
        move_evals = np.exp(move_evals)
        move_evals = move_evals * valid_cols
        move_evals /= np.sum(move_evals)

        move = np.random.choice(COLUMNS, p=move_evals)
        game.drop_piece(move)

        winner = game.get_winner()

    return training_data

def create_batch(num_episodes: int) -> list[tuple[np.longlong, np.longlong, np.bool, np.long]]:
    training_data = []
    for _ in range(num_episodes):
        training_data.extend(do_episode())
    return training_data

def save_training_data(training_data: list[list[tuple[np.longlong, np.longlong, np.bool, np.long]]]):
    np.save(path / "nnue" / "data" / "training_data.npy", training_data)

def load_training_data() -> list[list[tuple[np.longlong, np.longlong, np.bool, np.long]]]:
    return np.load(path / "nnue" / "data" / "training_data.npy", allow_pickle=True)

if __name__ == "__main__":
    start_time = time.time()
    
    current_batches = []
    if Path.exists(path / "nnue" / "data" / "training_data.npy"):
        current_batches = load_training_data()
    training_data = create_batch(10)
    current_batches.extend(training_data)

    print(np.shape(current_batches))

    if len(current_batches) > 500000:
        current_batches = current_batches[-500000:]
    save_training_data(current_batches)

    print(f"Time: {time.time() - start_time}")