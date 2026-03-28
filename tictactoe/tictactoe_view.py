from enum import IntEnum
from random import random

import discord


class TicTacToeButton(discord.ui.Button['TicTacToeView']):
    def __init__(self, x: int, y: int) -> None:
        super().__init__(style=discord.ButtonStyle.secondary, label="\u200b", row=y)
        self.x = x
        self.y = y

    async def callback(self, interaction: discord.Interaction) -> None:
        assert self.view is not None
        view: TicTacToeView = self.view

        user_id = interaction.user.id

        if user_id not in (view.player_X, view.player_O):
            await interaction.response.send_message(content="You are not in the game", ephemeral=True)
            return

        if user_id != view.get_current_player_id():
            await interaction.response.send_message(content="It is not your turn", ephemeral=True)
            return

        state = view.board[self.y][self.x]
        if state in (Symbol.X, Symbol.O):
            return

        if view.current_player == Symbol.X:
            self.style = discord.ButtonStyle.danger
            self.label = 'X'
            self.disabled = True
            view.place_symbol(self.x, self.y, Symbol.X)
            view.current_player = Symbol.O
            content = f"It is now <@{view.get_current_player_id()}>'s turn"
        elif view.current_player == Symbol.O:
            self.style = discord.ButtonStyle.success
            self.label = 'O'
            self.disabled = True
            view.place_symbol(self.x, self.y, Symbol.O)
            view.current_player = Symbol.X
            content = f"It is now <@{view.get_current_player_id()}>'s turn"

        winner = view.winner
        if winner is not None:
            content = view.get_winner_message()

            view.stop_game()
            view.stop()

        await interaction.response.edit_message(content=content, view=view)

class Symbol(IntEnum):
    X = -1
    EMPTY = 0
    O = 1
    TIE = 2

class TicTacToeView(discord.ui.View):
    def __init__(self, player_X_id: int, player_O_id: int) -> None:
        super().__init__()

        self.current_player = Symbol.X
        self.player_X = player_X_id
        self.player_O = player_O_id
        self.winner = None

        self.board = [[Symbol.EMPTY] * 3 for _ in range(3)]
        
        for x in range(3):
            for y in range(3):
                self.add_item(TicTacToeButton(x, y))

        # Bot First Move
        if self.player_X == -1:
            self.random_move()
    
    def get_current_player_id(self) -> Symbol:
        if self.current_player == Symbol.X:
            return self.player_X
        return self.player_O
    
    def place_symbol(self, x: int, y: int, symbol: Symbol) -> None:
        self.board[y][x] = symbol
        self.winner = self.check_for_win()

    def stop_game(self) -> None:
        for button in self.children:
            button.disabled = True

    def check_for_win(self) -> Symbol | None:
        for row in self.board:
            if row[0] == row[1] == row[2] != Symbol.EMPTY:
                return row[0]

        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != Symbol.EMPTY:
                return self.board[0][col]

        if self.board[0][0] == self.board[1][1] == self.board[2][2] != Symbol.EMPTY:
            return self.board[0][0]

        if self.board[0][2] == self.board[1][1] == self.board[2][0] != Symbol.EMPTY:
            return self.board[0][2]
        
        for row in self.board:
            if Symbol.EMPTY in row:
                return None

        return Symbol.TIE
    
    def get_winner_id(self) -> int:
        if self.winner == Symbol.X:
            return self.player_X
        if self.winner == Symbol.O:
            return self.player_O
        return None
    
    def get_loser_id(self) -> int:
        if self.winner == Symbol.X:
            return self.player_O
        if self.winner == Symbol.O:
            return self.player_X
        return None
    
    def get_winner_message(self) -> str:
        if self.winner == Symbol.TIE:
            return f"<@{self.player_X}> and <@{self.player_O}> tied!"
        return f"<@{self.get_winner_id()}> won against <@{self.get_loser_id()}>!"

    def random_move(self) -> tuple[int, int] | None:
        empty_squares = []
        for x in range(3):
            for y in range(3):
                if self.board[y][x] == Symbol.EMPTY:
                    empty_squares.append((x, y))

        if empty_squares:
            x, y = random.choice(empty_squares)
            self.place_symbol(x, y, self.current_player)
            for button in self.children:
                if button.x == x and button.y == y:
                    if self.current_player == Symbol.X:
                        button.style = discord.ButtonStyle.danger
                        button.label = 'X'
                        button.disabled = True
                    elif self.current_player == Symbol.O:
                        button.style = discord.ButtonStyle.success
                        button.label = 'O'
                        button.disabled = True

            self.current_player = -self.current_player
            return (x, y)

        return None