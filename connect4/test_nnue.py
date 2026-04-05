from pathlib import Path
import time

import numpy as np
import torch
from nnue.connect4_game import Color, Connect4Game
from nnue.connect4_minimax import Connect4Minimax
from nnue.connect4_nnue import Connect4NNUE

def test_nnue() -> None:
    game = Connect4Game()
    game.nnue_wrapper.load_model(model)
    res1 = game.nnue_wrapper.nn.forward(torch.tensor(game.red_bitboard, dtype=torch.long, device="cuda"), \
                                        torch.tensor(game.yellow_bitboard, dtype=torch.long, device="cuda"), \
                                        torch.tensor(Color.YELLOW, dtype=torch.long, device="cuda"))
    print(res1)
    res1 = game.nnue_wrapper.accumulator_forward(-Color.YELLOW)
    print("wrapper: ", res1)
    # print(nn.red_accumulator.weight[:, 35])
    # print(nn.red_accumulator_features)
    game.nnue_wrapper.nn.accumulator_add(2, 2, Color.YELLOW)
    res2 = game.nnue_wrapper.nn.accumulator_forward(-Color.YELLOW)
    print(res2)
    res2 = game.nnue_wrapper.accumulator_forward(-Color.YELLOW)
    print("wrapper: ", res2)
    # print(nn.red_accumulator_features)

def test_nnue_wrapper() -> None:
    game = Connect4Game()
    game.nnue_wrapper.load_model(model)
    res1 = game.nnue_wrapper.evaluate_board(game.red_bitboard, game.yellow_bitboard, Color.YELLOW)
    print("wrapper: ", res1)
    game.nnue_wrapper.accumulator_add(5, 2, Color.YELLOW)
    res2 = game.nnue_wrapper.accumulator_forward(-Color.YELLOW)
    print("wrapper: ", res2)

def test_game() -> None:
    game = Connect4Game()
    game.load_model(model)
    game.print_bitboard()
    res1 = game.nnue_wrapper.evaluate_board(game.red_bitboard, game.yellow_bitboard, Color.RED)
    print("wrapper: ", res1)
    game.drop_piece_with_color(3, Color.RED)
    game.print_bitboard()
    res2 = game.nnue_wrapper.evaluate_board(game.red_bitboard, game.yellow_bitboard, Color.YELLOW)
    print("wrapper: ", res2)
    game.drop_piece_with_color(3, Color.YELLOW)
    game.print_bitboard()
    res2 = game.nnue_wrapper.evaluate_board(game.red_bitboard, game.yellow_bitboard, Color.RED)
    print("wrapper: ", res2)


def test_first_8_ply() -> None:
    game = Connect4Game()
    game.load_model(model)
    first_8_ply = np.load(path / "nnue" / "data" / "nnue" / "data" / "first_8_ply.npy", allow_pickle=True)
    for i, (red_bitboard, yellow_bitboard, player, eval) in enumerate(first_8_ply):
        game.red_bitboard = red_bitboard
        game.yellow_bitboard = yellow_bitboard
        game.player = player
        pieces = bin(red_bitboard).count("1") + bin(yellow_bitboard).count("1")
        if pieces != 8:
            game.print_bitboard()
            print("Player: ", player)
            print("Eval: ", eval)
            print()

def test_eval() -> None:
    game = Connect4Game()
    game.load_model(model)
    game.print_bitboard()
    print(game.evaluate_board_reset())
    fresh_evals = []
    nnue_evals = []
    for col in range(7):
        move = game.drop_piece(col)
        fresh_evals.append(game.evaluate_board_reset())
        game.remove_piece(move[0], move[1])
    for col in range(7):
        move = game.drop_piece(col)
        nnue_evals.append(game.evaluate_board())
        game.remove_piece(move[0], move[1])
    print("Fresh evals: ", fresh_evals)
    print("NNUE evals: ", nnue_evals)

def test_eval2() -> None:
    game = Connect4Game()
    game.load_model(model)
    move = game.drop_piece(3)
    game.print_bitboard()
    print(game.evaluate_board_reset())
    fresh_evals = []
    nnue_evals = []
    for col in range(7):
        move = game.drop_piece(col)
        fresh_evals.append(game.evaluate_board_reset())
        game.remove_piece(move[0], move[1])
    for col in range(7):
        move = game.drop_piece(col)
        nnue_evals.append(game.evaluate_board())
        game.remove_piece(move[0], move[1])
    print("Fresh evals: ", fresh_evals)
    print("NNUE evals: ", nnue_evals)

def test_eval_on_moves(moves: str) -> None:
    game = Connect4Game()
    game.load_model(model)

    # Play moves
    for move in moves:
        game.drop_piece(int(move) - 1)

    game.print_bitboard()
    print(game.evaluate_board_reset())

    fresh_evals = []
    nnue_evals = []
    for col in range(7):
        move = game.drop_piece(col)
        fresh_evals.append(game.evaluate_board_reset())
        game.remove_piece(move[0], move[1])
    for col in range(7):
        move = game.drop_piece(col)
        nnue_evals.append(game.evaluate_board())
        game.remove_piece(move[0], move[1])
    print("Fresh evals: ", fresh_evals)
    print("NNUE evals: ", nnue_evals)

path = Path(__file__).parent
model = path / "nnue" / "models" / "best.pt"
# test_nnue()
# test_nnue_wrapper()
# test_game()
# test_first_8_ply()
# test_eval()
# test_eval2()
# test_eval_on_moves("4")
# test_eval_on_moves("41")
# test_eval_on_moves("42")
# test_eval_on_moves("43")
# test_eval_on_moves("44")
# test_eval_on_moves("45")
# test_eval_on_moves("46")
# test_eval_on_moves("47")