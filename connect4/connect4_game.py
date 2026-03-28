from enum import IntEnum

import numpy as np


class Color(IntEnum):
    RED = -1
    EMPTY = 0
    YELLOW = 1
    RANDOM = 2

class Connect4Game:
    rows = 6
    cols = 7

    def get_empty_board() -> np.array:
        return np.zeros(shape=(6,7))
    
    def display_board(board: np.array) -> None:
        print("Connect 4 Board")
        for i, row in enumerate(board):
            print(i, end=" ")
            for col in row:
                if col == Color.RED:
                    print("R", end=" ")
                elif col == Color.YELLOW:
                    print("Y", end=" ")
                else:
                    print(".", end=" ")
            print()
        print("  0 1 2 3 4 5 6")

    def get_emoji(color: Color) -> str:
        match color:
            case Color.RED:
                return "🔴"
            case Color.YELLOW:
                return "🟡"
            case Color.EMPTY:
                return "⚪"
            case _:
                raise Exception("Invalid Color")

    def get_emoji_board(board: np.array) -> str:
        emoji_board = ""
        for row in board:
            for col in row:
                if col == Color.RED:
                    emoji_board += "🔴"
                elif col == Color.YELLOW:
                    emoji_board += "🟡"
                else:
                    emoji_board += "⚪"
            emoji_board += "\n"
        return emoji_board
    
    def drop_piece_get_board(board: np.array, col: int, color: Color) -> tuple[np.array, tuple[int, int], Color]:
        """
        Not Inplace

        Returns the new board, the row and col of the piece dropped, and the color of the next player
        """
        assert col >= 0 and col < 7, "Invalid Column"
        assert color in (Color.RED, Color.YELLOW), "Invalid Color"
        assert board[0][col] == Color.EMPTY, "Column is full"
        new_board = Connect4Game.get_empty_board()
        for row in reversed(range(6)):
            if new_board[row][col] == Color.EMPTY:
                new_board[row][col] = color
                return (new_board, (row, col), -color)
        raise Exception("Invalid Move")

    def drop_piece(board: np.array, col: int, color: Color) -> tuple[tuple[int, int], Color]:
        """Returns the row and col of the piece dropped, and the color of the next player"""
        assert col >= 0 and col < 7, "Invalid Column"
        assert color in (Color.RED, Color.YELLOW), "Invalid Color"
        assert board[0][col] == Color.EMPTY, "Column is full"
        for row in reversed(range(6)):
            if board[row][col] == Color.EMPTY:
                board[row][col] = color
                return ((row, col), -color)
        raise Exception("Invalid Move")
    
    def get_valid_cols(board: np.array) -> list[int]:
        return [i for i in range(7) if board[0][i] == Color.EMPTY]
    
    def is_column_full(board: np.array, col: int) -> bool:
        return board[0][col] != Color.EMPTY

    def get_game_win(board: np.array, row: int, col: int) -> Color | None:
        """
        returns -1 if red wins, 1 if yellow wins, 1e-4 if tie
        """
        assert board[row][col] in (Color.RED, Color.YELLOW), "Invalid Space Selected To Check For Win"

        color = board[row][col]

        # Check row
        count = 0
        for c in range(7):
            if board[row][c] == color:
                count += 1
            else:
                count = 0
            if count == 4:
                return color
        
        # Check col
        count = 0
        for r in range(6):
            if board[r][col] == color:
                count += 1
            else:
                count = 0
            if count == 4:
                return color
            
        # Check Diagonals
        # Top Left to Bottom Right
        count = 0
        curr_row, curr_col = row - min(row, col), col - min(row, col)
        while curr_row < 6 and curr_col < 7:
            if board[curr_row][curr_col] == color:
                count += 1
            else: 
                count = 0
            if count == 4:
                return color
            curr_row += 1
            curr_col += 1

        # Bottom Left to Top Right
        count = 0
        row_from_bottom = Connect4Game.rows - row - 1
        curr_row, curr_col = row + min(row_from_bottom, col), col - min(row_from_bottom, col)
        while curr_row >= 0 and curr_col < 7:
            if board[curr_row][curr_col] == color:
                count += 1
            else:
                count = 0
            if count == 4:
                return color
            curr_row -= 1
            curr_col += 1

        # Check for tie
        if board.all():
            return 1e-4

        return None