import logging
from random import choice

import discord

from .tictactoe_view import Symbol, TicTacToeView

logger = logging.getLogger("cogs.tictactoe")


class ChooseSymbolButton(discord.ui.Button['TicTacToeChooseSymbolView']):
    def __init__(self, symbol: Symbol) -> None:
        if symbol == Symbol.X:
            self.symbol = symbol
            super().__init__(style=discord.ButtonStyle.danger, label="X", row=1)
        elif symbol == Symbol.O:
            self.symbol = symbol
            super().__init__(style=discord.ButtonStyle.success, label="O", row=1)

    async def callback(self, interaction: discord.Interaction) -> None:
        assert self.view is not None
        view: TicTacToeChooseSymbolView = self.view

        if interaction.user.id == view.player_1:
            view.player_1_choice = self.symbol
        elif interaction.user.id == view.player_2:
            view.player_2_choice = self.symbol
        else:
            await interaction.response.send_message(content="You are not in the game", ephemeral=True)
            return
        
        await interaction.response.defer(ephemeral=True)
        
        if view.player_1_choice is not None and view.player_2_choice is not None:
            X_player = view.determine_X_player(view.player_1_choice, view.player_2_choice)
            logger.info(f"<@{view.player_1}> chose {view.player_1_choice} and <@{view.player_2}> chose {view.player_2_choice}. <@{X_player}> goes first")
            O_player = view.player_1 if X_player == view.player_2 else view.player_2
            view.stop()
            await interaction.followup.edit_message(message_id=interaction.message.id, content=f"It is now <@{X_player}>'s turn", view=TicTacToeView(X_player, O_player))

    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        logger.error(error)

class TicTacToeChooseSymbolView(discord.ui.View):
    def __init__(self, player_1_id: int, player_2_id: int) -> None:
        super().__init__()
        self.player_1 = player_1_id
        self.player_2 = player_2_id
        self.player_1_choice = None
        self.player_2_choice = None
        self.add_item(ChooseSymbolButton(Symbol.X))
        self.add_item(ChooseSymbolButton(Symbol.O))
    
    def determine_X_player(self, player_1_choice: Symbol, player_2_choice: Symbol) -> int:
        if player_1_choice == None or player_2_choice == None:
            return choice([self.player_1, self.player_2])
        if player_1_choice == Symbol.X and player_2_choice == Symbol.O:
            return self.player_1
        if player_1_choice == Symbol.O and player_2_choice == Symbol.X:
            return self.player_2
        return choice([self.player_1, self.player_2])