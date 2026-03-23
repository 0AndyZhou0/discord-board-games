from enum import IntEnum
from random import choice, shuffle


class Symbol(IntEnum):
    EMPTY = 0
    X = -1
    O = 1
    TIE = 2

class TicTacToeBot:

    def minimax(board: list[list[Symbol]], depth: int, alpha: int, beta: int, maximizing_x: bool) -> int:
        if TicTacToeBot.check_for_win(board) is not None:
            return TicTacToeBot.evaluate_board(board)
        
        if maximizing_x:
            best_alpha = float('-inf')
            for (x, y) in TicTacToeBot.find_empty_cells(board):
                board[x][y] = Symbol.X
                best_alpha = max(best_alpha, TicTacToeBot.minimax(board, depth+1, alpha, beta, False))
                board[x][y] = Symbol.EMPTY
                alpha = max(alpha, best_alpha)
                if alpha >= beta:
                    break
            return best_alpha
        else:  # noqa: RET505
            best_beta = float('inf')
            for (x, y) in TicTacToeBot.find_empty_cells(board):
                board[x][y] = Symbol.O
                best_beta = min(best_beta, TicTacToeBot.minimax(board, depth+1, alpha, beta, True))
                board[x][y] = Symbol.EMPTY
                beta = min(beta, best_beta)
                if alpha >= beta:
                    break
            return best_beta
    
    def find_best_move(board: list[list[Symbol]], symbol: Symbol) -> tuple[int, int]:
        if symbol == Symbol.X:
            best_value = float('-inf')
            best_move = None
            for (x, y) in TicTacToeBot.find_empty_cells(board):
                board[x][y] = Symbol.X
                value = TicTacToeBot.minimax(board, 0, float('-inf'), float('inf'), False)
                board[x][y] = Symbol.EMPTY
                if value > best_value:
                    best_value = value
                    best_move = (x, y)
            return best_move
        if symbol == Symbol.O:
            best_value = float('inf')
            best_move = None
            for (x, y) in TicTacToeBot.find_empty_cells(board):
                board[x][y] = Symbol.O
                value = TicTacToeBot.minimax(board, 0, float('-inf'), float('inf'), True)
                board[x][y] = Symbol.EMPTY
                if value < best_value:
                    best_value = value
                    best_move = (x, y)
            return best_move
        raise Exception("Invalid symbol")
    
    def find_best_move_first_weighted(board: list[list[Symbol]], symbol: Symbol) -> tuple[int, int]:
        if all(cell == Symbol.EMPTY for row in board for cell in row):
            return TicTacToeBot.pick_first_move_weighted()
        return TicTacToeBot.find_best_move(board, symbol)
    
    def pick_first_move_weighted() -> tuple[int, int]:
        choices = [
            # 8x corners
            (0, 0),
            (0, 2),
            (2, 0),
            (2, 2),
            (0, 0),
            (0, 2),
            (2, 0),
            (2, 2),
            # 4x centers
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            # 4x edges
            (0, 1),
            (2, 1),
            (1, 0),
            (1, 2)
        ]
        return choice(choices)
    
    def find_empty_cells(board: list[list[Symbol]]) -> list[tuple[int, int]]:
        empty_cells = []
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    empty_cells.append((i, j))
        shuffle(empty_cells)
        return empty_cells
    def check_for_win(board: list[list[Symbol]]) -> Symbol | None:
        for row in board:
            if row[0] == row[1] == row[2] != Symbol.EMPTY:
                return row[0]

        for col in range(3):
            if board[0][col] == board[1][col] == board[2][col] != Symbol.EMPTY:
                return board[0][col]

        if board[0][0] == board[1][1] == board[2][2] != Symbol.EMPTY:
            return board[0][0]

        if board[0][2] == board[1][1] == board[2][0] != Symbol.EMPTY:
            return board[0][2]
        
        for row in board:
            if Symbol.EMPTY in row:
                return None

        return Symbol.TIE

    def evaluate_board(board: list[list[Symbol]]) -> int:
        if TicTacToeBot.check_for_win(board) == Symbol.TIE:
            return 0
        if TicTacToeBot.check_for_win(board) == Symbol.X:
            return 1
        if TicTacToeBot.check_for_win(board) == Symbol.O:
            return -1
        raise Exception("Invalid board state")
