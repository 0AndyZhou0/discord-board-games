import ctypes
from pathlib import Path

import numpy as np
import torch

from nnue.connect4_game import Connect4Game
from get_training_from_cpp import TrainingData
from nnue.connect4_nnue_wrapper import Connect4NNUEWrapper
import logging
logging.basicConfig()
logger = logging.getLogger("cogs.connect4.nnue.train")

if __name__ == "__main__":
    MAX_TRAINING_POSITIONS = 20000000
    NUM_SAMPLES = 0

    # logger.setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)

    game = Connect4Game()

    path = Path(__file__).parent
    # training_data = torch.load(path / "nnue" / "data" / "training_data.pt")
    book_path = (path / "nnue" / "7x6.book").__bytes__()
    solver = ctypes.CDLL(path / "nnue" / "trainingData.so")
    solver.c_get_training_data.restype = ctypes.POINTER(TrainingData)
    TrainingDataPtr = ctypes.POINTER(TrainingData)

    training_data = []
    if Path.exists(path / "nnue" / "data" / "training_data.pt"):
        training_data = torch.load(path / "nnue" / "data" / "training_data.pt")

    nnue_wrapper = Connect4NNUEWrapper()
    if Path.exists(path / "nnue" / "models" / "best.pt"):
        logger.info("loading model")
        nnue_wrapper.load_model(path / "nnue" / "models" / "best.pt")
    
    iters = 1000

    for i in range(iters):
        logger.info(f"iter: {i}")

        logger.info("getting training data")
        TrainingDataPtr = solver.c_get_training_data(123456, book_path)
        new_training_data = TrainingDataPtr.contents.get_batch_tensors()

        # print(np.shape(training_data))
        if training_data:
            red_bitboards, yellow_bitboards, players, scores = training_data
            new_red_bitboards, new_yellow_bitboards, new_players, new_scores = new_training_data
            combined_red_bitboards = torch.cat((red_bitboards, new_red_bitboards), dim=0)
            combined_yellow_bitboards = torch.cat((yellow_bitboards, new_yellow_bitboards), dim=0)
            combined_players = torch.cat((players, new_players), dim=0)
            combined_scores = torch.cat((scores, new_scores), dim=0)
            training_data = (combined_red_bitboards, combined_yellow_bitboards, combined_players, combined_scores)
        else:
            training_data = new_training_data
        if len(training_data[0]) > MAX_TRAINING_POSITIONS:
            training_data = (training_data[0][-MAX_TRAINING_POSITIONS:], \
                            training_data[1][-MAX_TRAINING_POSITIONS:], \
                            training_data[2][-MAX_TRAINING_POSITIONS:], \
                            training_data[3][-MAX_TRAINING_POSITIONS:])

        logger.info(f"training data shape: {np.shape(training_data)}")

        for i in range(NUM_SAMPLES):
            logger.debug(f"Training data index: {i}")
            game.red_bitboard = training_data[0][i]
            game.yellow_bitboard = training_data[1][i]
            game.player = training_data[2][i]
            logger.debug(f"Board:\n{game.to_string()}")
            logger.debug(f"Player: {game.player}")
            logger.debug(f"Eval: {training_data[3][i]}")
            logger.debug("")
        
        logger.info("saving training data")
        torch.save(training_data, path / "nnue" / "data" / "training_data.pt")

        logger.info("training")
        nnue_wrapper.train(training_data, epochs=10, batch_size=2048, batch_count=100)

        # battle nnue against old nnue

        logger.info("saving model")
        nnue_wrapper.save_model(path / "nnue" / "models" / f"best.pt")