import discord

from .connect4_choose_color import Connect4ChooseColorView


class AcceptButton(discord.ui.Button["Connect4AcceptView"]):
    def __init__(self) -> None:
        super().__init__(style=discord.ButtonStyle.success, label="Accept", row=1)

    async def callback(self, interaction: discord.Interaction) -> None:
        assert self.view is not None
        view: Connect4AcceptView = self.view

        user_id = interaction.user.id

        if user_id != view.player_id:
            self.disabled = True
            view.stop()
            await interaction.response.edit_message(view=Connect4ChooseColorView(view.player_id, user_id))
        else:
            await interaction.response.send_message(content="You are already in the game", ephemeral=True)
            return

class Connect4AcceptView(discord.ui.LayoutView):
    def __init__(self, player_id: int) -> None:
        super().__init__()
        self.player_id = player_id
        self.add_item(discord.ui.ActionRow(AcceptButton()))