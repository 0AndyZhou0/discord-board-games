from enum import IntEnum

import discord


class TicTacToeButton(discord.ui.Button['TicTacToeView']):
    def __init__(self, x: int, y: int) -> None:
        super().__init__(style=discord.ButtonStyle.secondary, label="\u200b", row=y)
        self.x = x
        self.y = y

    async def callback(self, interaction: discord.Interaction) -> None:
        assert self.view is not None
        view: TicTacToeView = self.view

        if interaction.user.id not in (view.player_X, view.player_O):
            await interaction.response.send_message(content="You are not in the game", ephemeral=True)
            return

        if interaction.user.id != view.get_current_player_id():
            await interaction.response.send_message(content="It is not your turn", ephemeral=True)
            return

        state = view.board[self.y][self.x]
        if state in (Symbol.X, Symbol.O):
            return

        if view.current_player == Symbol.X:
            self.style = discord.ButtonStyle.danger
            self.label = 'X'
            self.disabled = True
            view.board[self.y][self.x] = Symbol.X
            view.current_player = Symbol.O
            content = f"It is now <@{view.get_current_player_id()}>'s turn"
        elif view.current_player == Symbol.O:
            self.style = discord.ButtonStyle.success
            self.label = 'O'
            self.disabled = True
            view.board[self.y][self.x] = Symbol.O
            view.current_player = Symbol.X
            content = f"It is now <@{view.get_current_player_id()}>'s turn"

        winner = view.check_for_win()
        if winner is not None:
            if winner == Symbol.X:
                content = f'<@{view.player_X}> won!'
            elif winner == Symbol.O:
                content = f'<@{view.player_O}> won!'
            else:
                content = "It's a tie!"

            for child in view.children:
                child.disabled = True

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

        self.board = [[Symbol.EMPTY] * 3 for _ in range(3)]
        
        for x in range(3):
            for y in range(3):
                self.add_item(TicTacToeButton(x, y))
    
    def get_current_player_id(self) -> Symbol:
        if self.current_player == Symbol.X:
            return self.player_X
        return self.player_O

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