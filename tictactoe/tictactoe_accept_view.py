import discord

from cogs_and_functions.tictactoe.tictactoe_choose_symbol_view import (
    TicTacToeChooseSymbolView,
)


class AcceptButton(discord.ui.Button['TicTacToeAcceptView']):
    def __init__(self) -> None:
        super().__init__(style=discord.ButtonStyle.success, label="Accept", row=1)

    async def callback(self, interaction: discord.Interaction) -> None:
        assert self.view is not None
        view: TicTacToeAcceptView = self.view

        user_id = interaction.user.id

        if user_id != view.player:
            self.disabled = True
            view.stop()
            await interaction.response.edit_message(content="Choose Symbol You Want To Play", view=TicTacToeChooseSymbolView(view.player, user_id))
        else:
            await interaction.response.send_message(content="You are already in the game", ephemeral=True)
            return

class TicTacToeAcceptView(discord.ui.View):
    def __init__(self, player_id: int) -> None:
        super().__init__()
        self.player = player_id
        self.add_item(AcceptButton())
