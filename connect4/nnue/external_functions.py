from ctypes import *
from pathlib import Path


path = Path(__file__).parent
book_path = (path / "7x6.book").__bytes__()
solver = CDLL(str(path) + "/trainingData.so")
solver.c_get_move_evals.restype = POINTER(c_int * 7)

def get_eval(moves: str) -> int:
    """score of current player"""
    return solver.c_get_eval(moves.encode(), book_path)

def get_move_evals(moves: str) -> list[int]:
    """value of each move for current player"""
    res = solver.c_get_move_evals(moves.encode(), book_path)
    return [i for i in res.contents]

MIN_SCORE = -(7*6) / 2 + 3;
MAX_SCORE = (7 * 6 + 1) / 2 - 3;
