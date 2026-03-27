from connect4_game import Color, Connect4Game


def test_invalid_moves() -> None:
    board = Connect4Game.get_empty_board()
    try:
        for _ in range(7):
            Connect4Game.drop_piece(board, 0, Connect4Game.Color.RED)
        assert False, "Expected Invalid Move"
    except Exception:
        pass
    try:
        for _ in range(7):
            Connect4Game.drop_piece(board, 0, Connect4Game.Color.YELLOW)
        assert False, "Expected Invalid Move"
    except Exception:
        pass

def test_moves() -> None:
    board = Connect4Game.get_empty_board()
    Connect4Game.display_board(board)
    import random
    player = Color.RED
    for _ in range(20):
        move, player = Connect4Game.drop_piece(board, random.choice(Connect4Game.get_valid_moves(board)), player)
        Connect4Game.display_board(board)
        win = Connect4Game.get_game_win(board, *move)
        if win:
            print(f"Player {player} wins")
            break