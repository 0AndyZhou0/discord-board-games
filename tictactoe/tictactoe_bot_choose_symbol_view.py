from random import choice

import discord

from .tictactoe_bot_view import Bot_Mode, Symbol, TicTacToeBotView


class ChooseSymbolButton(discord.ui.Button['TicTacToeBotChooseSymbolView']):
    def __init__(self, symbol: Symbol) -> None:
        if symbol == Symbol.X:
            self.symbol = symbol
            super().__init__(style=discord.ButtonStyle.danger, label="❌", row=1)
        elif symbol == Symbol.O:
            self.symbol = symbol
            super().__init__(style=discord.ButtonStyle.success, label="⭕", row=1)
        elif symbol == Symbol.RANDOM:
            self.symbol = symbol
            super().__init__(style=discord.ButtonStyle.secondary, label="🔄", row=1)

    async def callback(self, interaction: discord.Interaction) -> None:
        assert self.view is not None
        view: TicTacToeBotChooseSymbolView = self.view

        user_id = interaction.user.id

        if user_id != view.player_id:
            interaction.response.send_message(content="You are not in the game", ephemeral=True)
            return
        
        view.player_choice = self.symbol
        X_player = view.determine_X_player(view.player_choice)

        if X_player == user_id:
            view.stop()
            await interaction.response.edit_message(content=f"It is now <@{user_id}>'s turn", view=TicTacToeBotView(user_id, -1, view.mode))
        else:
            view.stop()
            await interaction.response.edit_message(content=f"It is now <@{user_id}>'s turn", view=TicTacToeBotView(-1, user_id, view.mode))

class TicTacToeBotChooseSymbolView(discord.ui.View):
    def __init__(self, player_id: int, mode: Bot_Mode) -> None:
        super().__init__()
        self.player_id = player_id
        self.player_choice = None
        self.mode = mode
        self.add_item(ChooseSymbolButton(Symbol.X))
        self.add_item(ChooseSymbolButton(Symbol.O))
        self.add_item(ChooseSymbolButton(Symbol.RANDOM))
    
    def determine_X_player(self, player_choice: Symbol) -> int:
        match player_choice:
            case Symbol.X:
                return self.player_id
            case Symbol.O:
                return -1
            case Symbol.RANDOM:
                return choice([self.player_id, -1])