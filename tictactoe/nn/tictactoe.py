import numpy as np

from .board import Symbol, TicTacToeBoard


class TicTacToe:
    def get_empty_board() -> np.array:
        board = TicTacToeBoard()
        return np.array(board.board)
    
    def get_action_size() -> int:
        return 10
    
    def get_next_board(board: np.array, player: Symbol, action: int) -> np.array:
        new_board = TicTacToeBoard()
        new_board.board = np.copy(board)
        new_board.play_move_flattened(action, player)
        return (new_board.board, -player)
    
    def get_empty_squares(board: np.array) -> list[tuple[int, int]]:
        new_board = TicTacToeBoard()
        new_board.board = np.copy(board)
        return new_board.get_empty_squares()
    
    def get_empty_squares_mask(board: np.array) -> list[int]:
        new_board = TicTacToeBoard()
        new_board.board = np.copy(board)
        return new_board.get_empty_squares_mask()
    
    def get_game_ended(board: np.array) -> int | None:
        new_board = TicTacToeBoard()
        new_board.board = np.copy(board)
        return new_board.get_game_ended()
    
    def get_current_player(board: np.array) -> Symbol:
        new_board = TicTacToeBoard()
        new_board.board = np.copy(board)
        return new_board.get_current_player()
    
    def get_canonical_board(board: np.array, player: Symbol) -> np.array:
        return player * np.array(board)
    
    def to_string(board: np.array) -> str:
        b = ""
        for i in range(3):
            for j in range(3):
                if board[i][j] == Symbol.X:
                    b += "X"
                elif board[i][j] == Symbol.O:
                    b += "O"
                else:
                    b += "."
            b += "\n"
        return b

    def __str__(self) -> str:
        b = ""
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == Symbol.X:
                    b += "X"
                elif self.board[i][j] == Symbol.O:
                    b += "O"
                else:
                    b += "."
            b += "\n"
        return b