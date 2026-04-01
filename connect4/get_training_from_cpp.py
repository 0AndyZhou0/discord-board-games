
import numpy as np
import torch

from nnue.connect4_game import Connect4Game

from nnue.external_functions import get_move_evals, get_eval
from pathlib import Path
import time

import ctypes

class TrainingData(ctypes.Structure):
    _fields_ = [
        ('size', ctypes.c_int),
        ('red_bitboards', ctypes.POINTER(ctypes.c_int64)),
        ('yellow_bitboards', ctypes.POINTER(ctypes.c_int64)),
        ('players', ctypes.POINTER(ctypes.c_int)),
        ('scores', ctypes.POINTER(ctypes.c_int))
    ]

    def get_batch_tensors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        red_bitboards = torch.from_numpy(np.ctypeslib.as_array(self.red_bitboards, shape=(self.size, 1)))
        yellow_bitboards = torch.from_numpy(np.ctypeslib.as_array(self.yellow_bitboards, shape=(self.size, 1)))
        players = torch.from_numpy(np.ctypeslib.as_array(self.players, shape=(self.size, 1)))
        scores = torch.from_numpy(np.ctypeslib.as_array(self.scores, shape=(self.size, 1)))
        return red_bitboards, yellow_bitboards, players, scores
    
    def get_batch_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        red_bitboards = np.ctypeslib.as_array(self.red_bitboards, shape=(self.size, 1))
        yellow_bitboards = np.ctypeslib.as_array(self.yellow_bitboards, shape=(self.size, 1))
        players = np.ctypeslib.as_array(self.players, shape=(self.size, 1))
        scores = np.ctypeslib.as_array(self.scores, shape=(self.size, 1))
        return red_bitboards, yellow_bitboards, players, scores

if __name__ == "__main__":
    path = Path(__file__).parent
    book_path = (path / "nnue" / "7x6.book").__bytes__()
    solver = ctypes.CDLL(path / "nnue" / "trainingData.so")
    solver.c_get_training_data.restype = ctypes.POINTER(TrainingData)

    TrainingDataPtr = ctypes.POINTER(TrainingData)
    TrainingDataPtr = solver.c_get_training_data(500000, book_path)
    print("TrainingDataPtr: ", TrainingDataPtr)
    np.save(path / "nnue" / "data" / "training_data.npy", TrainingDataPtr.contents.get_batch_numpy())
    # torch.save(TrainingDataPtr.contents.get_batch_tensors(), path / "nnue" / "data" / "training_data.pt")