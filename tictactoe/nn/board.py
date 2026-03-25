from enum import IntEnum

import numpy as np


class Symbol(IntEnum):
    """
    Fields:
        EMPTY = 0
        X = -1
        O = 1
        TIE = 2
    """
    EMPTY = 0
    X = -1
    O = 1
    TIE = 2

class TicTacToeBoard:
    def __init__(self) -> None:
        # self.board = [[Symbol.EMPTY for _ in range(3)] for _ in range(3)]
        self.board = np.zeros((3, 3))
    
    def __getitem__(self, row: int) -> list[Symbol]:
        return self.board[row]
    
    def get_current_player(self) -> Symbol:
        count_x = 0
        count_o = 0
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == Symbol.X:
                    count_x += 1
                elif self.board[i][j] == Symbol.O:
                    count_o += 1
        if count_x > count_o:
            return Symbol.O
        return Symbol.X
    
    def get_empty_squares(self) -> list[tuple[int, int]]:
        empty_squares = []
        for x in range(3):
            for y in range(3):
                if self.board[x][y] == Symbol.EMPTY:
                    empty_squares.append((x, y))
        return empty_squares
    
    def get_empty_squares_mask(self) -> np.array:
        empty_squares = np.zeros(9)
        for x in range(3):
            for y in range(3):
                if self.board[x][y] == Symbol.EMPTY:
                    empty_squares[x * 3 + y] = 1
        return empty_squares

    def get_winner(self) -> Symbol | None:
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != Symbol.EMPTY:
                return self.board[i][0]
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != Symbol.EMPTY:
                return self.board[0][i]
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != Symbol.EMPTY:
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != Symbol.EMPTY:
            return self.board[0][2]
        return None
    
    def check_for_win(self, player: Symbol) -> bool:
        return self.get_winner() == player
    
    def get_game_ended(self) -> int | None:
        """
        Returns:
            -1 if X won, 
            1e-4 if tie, 
            1 if O won, 
            None if game is not over, 
        """
        winner = self.get_winner()
        if winner is None and len(self.get_empty_squares()) == 0:
            return 1e-4
        return winner
    
    def is_win_for(self, symbol: Symbol) -> bool:
        return self.get_winner() == symbol
    
    def play_move_flattened(self, position: int, symbol: Symbol) -> None:
        x = position // 3
        y = position % 3
        self.play_move((x, y), symbol)

    def play_move(self, position: tuple[int, int], symbol: Symbol) -> None:
        (x, y) = position
        if self.board[x][y] != Symbol.EMPTY:
            print(self.board)
            print(position)
        assert self.board[x][y] == Symbol.EMPTY
        self.board[x][y] = symbol

    def __str__(self) -> str:
        board = ""
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == Symbol.X:
                    board += "X"
                elif self.board[i][j] == Symbol.O:
                    board += "O"
                else:
                    board += "."
            board += "\n"
        return board
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TicTacToeBoard):
            return False
        return self.board == other.board
    
    def __hash__(self) -> int:
        return hash(str(self))
    