import ctypes
from pathlib import Path

import numpy as np
import torch

from get_training_from_cpp import TrainingData
from nnue.connect4_nnue_wrapper import Connect4NNUEWrapper

if __name__ == "__main__":
    path = Path(__file__).parent
    # training_data = torch.load(path / "nnue" / "data" / "training_data.pt")
    book_path = (path / "nnue" / "7x6.book").__bytes__()
    solver = ctypes.CDLL(path / "nnue" / "trainingData.so")
    solver.c_get_training_data.restype = ctypes.POINTER(TrainingData)
    TrainingDataPtr = ctypes.POINTER(TrainingData)

    training_data = []
    if Path.exists(path / "nnue" / "data" / "training_data.pt"):
        training_data = torch.load(path / "nnue" / "data" / "training_data.pt")

    iters = 1000

    for i in range(iters):
        print("iter: ", i)

        nnue_wrapper = Connect4NNUEWrapper()
        if Path.exists(path / "nnue" / "models" / "best.pt"):
            print("loading model")
            nnue_wrapper.load_model(path / "nnue" / "models" / "best.pt")

        print("getting training data")
        TrainingDataPtr = solver.c_get_training_data(50000, book_path)
        new_training_data = TrainingDataPtr.contents.get_batch_tensors()

        # print(np.shape(training_data))
        if training_data:
            red_bitboards, yellow_bitboards, players, scores = training_data
            new_red_bitboards, new_yellow_bitboards, new_players, new_scores = new_training_data
            print(np.shape(red_bitboards), np.shape(new_red_bitboards))
            combined_red_bitboards = torch.cat((red_bitboards, new_red_bitboards), dim=0)
            combined_yellow_bitboards = torch.cat((yellow_bitboards, new_yellow_bitboards), dim=0)
            combined_players = torch.cat((players, new_players), dim=0)
            combined_scores = torch.cat((scores, new_scores), dim=0)
            training_data = (combined_red_bitboards, combined_yellow_bitboards, combined_players, combined_scores)
        else:
            training_data = new_training_data
        if len(training_data[0]) > 2000000:
            training_data = (training_data[0][-2000000:], training_data[1][-2000000:], training_data[2][-2000000:], training_data[3][-2000000:])
        print(np.shape(training_data))
        print("saving training data")
        torch.save(training_data, path / "nnue" / "data" / "training_data.pt")

        print("training")
        nnue_wrapper.train(training_data, epochs=10, batch_size=2048)

        # battle nnue against old nnue

        print("saving model")
        nnue_wrapper.save_model(path / "nnue" / "models" / f"best.pt")