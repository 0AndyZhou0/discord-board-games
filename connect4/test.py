from pathlib import Path
import time

import numpy as np
import torch
from nnue.connect4_game import Color, Connect4Game
from nnue.connect4_minimax import Connect4Minimax
from nnue.connect4_nnue import Connect4NNUE

def game_test() -> None:
    game = Connect4Game()
    game.load_model(model)
    # board = np.array([
    #     [Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY],
    #     [Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY],
    #     [Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY],
    #     [Color.EMPTY, Color.EMPTY, Color.YELLOW, Color.RED, Color.YELLOW, Color.YELLOW, Color.EMPTY],
    #     [Color.EMPTY, Color.EMPTY, Color.RED, Color.YELLOW, Color.RED, Color.RED, Color.EMPTY],
    #     [Color.EMPTY, Color.EMPTY, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.RED, Color.EMPTY]
    # ])
    # board = np.array([
    #     [Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY],
    #     [Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY],
    #     [Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY, Color.EMPTY],
    #     [Color.EMPTY, Color.EMPTY, Color.RED, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.EMPTY],
    #     [Color.EMPTY, Color.EMPTY, Color.RED, Color.YELLOW, Color.RED, Color.RED, Color.EMPTY],
    #     [Color.EMPTY, Color.EMPTY, Color.YELLOW, Color.RED, Color.YELLOW, Color.RED, Color.EMPTY]
    # ])
    # red_bitboard, yellow_bitboard = game.to_bitboards(board)
    # game.red_bitboard = red_bitboard
    # game.yellow_bitboard = yellow_bitboard

    game.print_bitboard()
    result = Connect4Minimax.minimax(game, Color.RED, 4)
    print(result)
    for col in range(7):
        move = game.drop_piece_with_color(col, Color.RED)
        result = Connect4Minimax.minimax(game, Color.YELLOW, 3)
        game.remove_piece(move[0], move[1], Color.RED)
        print(f"Move: {col}, Result: {result}")

    # for col in range(7):
    #     move = game.drop_piece(col, Color.RED)
    #     result = Connect4Minimax.minimax(game, Color.YELLOW, 2)
    #     game.remove_piece(move[0], move[1], Color.RED)
    #     print(f"Move: {col}, Result: {result}")

    # Connect4Minimax.minimax(game, Color.RED, 5)

    move = game.drop_piece_with_color(1, Color.RED)
    move = game.drop_piece_with_color(2, Color.YELLOW)
    move = game.drop_piece_with_color(3, Color.RED)
    move = game.drop_piece_with_color(4, Color.YELLOW)
    move = game.drop_piece_with_color(5, Color.RED)
    move = game.drop_piece_with_color(6, Color.YELLOW)
    game.print_bitboard()
    result = Connect4Minimax.minimax(game, Color.RED, 3)
    print(f"Result: {result}")
    move = game.drop_piece_with_color(1, Color.RED)
    move = game.drop_piece_with_color(2, Color.YELLOW)
    move = game.drop_piece_with_color(3, Color.RED)
    move = game.drop_piece_with_color(4, Color.YELLOW)
    move = game.drop_piece_with_color(5, Color.RED)
    move = game.drop_piece_with_color(6, Color.YELLOW)
    move = game.drop_piece_with_color(2, Color.RED)
    move = game.drop_piece_with_color(4, Color.YELLOW)
    move = game.drop_piece_with_color(6, Color.RED)
    move = game.drop_piece_with_color(5, Color.YELLOW)
    move = game.drop_piece_with_color(6, Color.RED)
    move = game.drop_piece_with_color(2, Color.YELLOW)
    move = game.drop_piece_with_color(4, Color.RED)
    move = game.drop_piece_with_color(2, Color.YELLOW)
    game.print_bitboard()
    result = Connect4Minimax.minimax(game, Color.RED, 3)
    print(f"NNUE Result: {result}")

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

def test_bad_dataset() -> None:
    game = Connect4Game()
    game.load_model(model)
    bad_dataset = np.load(path / "nnue" / "data" / "bad_dataset.npy", allow_pickle=True)
    for i, (red_bitboard, yellow_bitboard, player, eval) in enumerate(bad_dataset[0:]):
        game.red_bitboard = red_bitboard
        game.yellow_bitboard = yellow_bitboard
        game.player = player
        game.print_bitboard()
        print("Player: ", player)
        print("Eval: ", eval)
        print()
        if i > 10:
            break
        # pieces = bin(red_bitboard).count("1") + bin(yellow_bitboard).count("1")
        # if pieces <= 8:
        #     game.print_bitboard()
        #     print("Player: ", player)
        #     print("Eval: ", eval)
        #     print()

def fix_bad_dataset() -> None:
    bad_dataset = np.load(path / "nnue" / "data" / "nnue" / "data" / "bad_dataset.npy", allow_pickle=True)
    fixed_dataset = []
    for i, (red_bitboard, yellow_bitboard, player, eval) in enumerate(bad_dataset):
        if eval == 0:
            eval = -1
        fixed_dataset.append((red_bitboard, yellow_bitboard, player, eval))
    np.save(path / "nnue" / "data" / "bad_dataset_fixed.npy", np.array(fixed_dataset))

def test_fixed_bad_dataset() -> None:
    game = Connect4Game()
    game.load_model(model)
    bad_dataset = np.load(path / "nnue" / "data" / "bad_dataset_fixed.npy", allow_pickle=True)
    for i, (red_bitboard, yellow_bitboard, player, eval) in enumerate(bad_dataset):
        game.red_bitboard = red_bitboard
        game.yellow_bitboard = yellow_bitboard
        game.player = player
        game.print_bitboard()
        print("Player: ", player)
        print("Eval: ", eval)
        print()
        if i > 10:
            break

path = Path(__file__).parent
model = path / "nnue" / "models" / "best.pt"
# test_nnue()
# test_nnue_wrapper()
# test_game()
# game_test()
# test_first_8_ply()
# test_bad_dataset()
# fix_bad_dataset()
# test_fixed_bad_dataset()