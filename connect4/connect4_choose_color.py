import logging
from random import choice

import discord

from .connect4_game import Color
from .connect4_view import Connect4View

logger = logging.getLogger("cogs.connect4")


class ChooseColorButton(discord.ui.Button['Connect4ChooseColorView']):
    def __init__(self, color: Color) -> None:
        match color:
            case Color.RED:
                emoji = "🔴"
            case Color.YELLOW:
                emoji = "🟡"
            case Color.RANDOM:
                emoji = "🎲"
            case _:
                raise Exception("Invalid Color")
        super().__init__(style=discord.ButtonStyle.primary, label=emoji, custom_id=color.name)
        self.symbol = color

    async def callback(self, interaction: discord.Interaction) -> None:
        assert self.view is not None
        view = self.view

        user_id = interaction.user.id

        if user_id not in (view.player_1_id, view.player_2_id):
            interaction.response.send_message(content="You are not in the game", ephemeral=True)
            return
        
        if user_id == view.player_1_id:
            view.player_1_choice = self.symbol
        if user_id == view.player_2_id:
            view.player_2_choice = self.symbol

        if view.player_1_choice is not None and view.player_2_choice is not None:
            red_player = view.determine_red_player(view.player_1_choice, view.player_2_choice)
            logger.info(f"<@{view.player_1_id}> chose {view.player_1_choice} and <@{view.player_2_id}> chose {view.player_2_choice}. <@{red_player}> goes first")
            yellow_player = view.player_1_id if red_player == view.player_2_id else view.player_2_id
            await interaction.response.edit_message(view=Connect4View(red_player, yellow_player))

        await interaction.response.defer(ephemeral=True)


class Connect4ChooseColorView(discord.ui.LayoutView):
    def __init__(self, player_1_id: int, player_2_id: int) -> None:
        super().__init__()
        self.player_1_id = player_1_id
        self.player_2_id = player_2_id
        self.player_1_choice = None
        self.player_2_choice = None

        action_row = discord.ui.ActionRow()
        action_row.add_item(ChooseColorButton(Color.RED))
        action_row.add_item(ChooseColorButton(Color.YELLOW))
        action_row.add_item(ChooseColorButton(Color.RANDOM))
        self.add_item(action_row)
    
    def determine_red_player(self, player_1_choice: Color, player_2_choice: Color) -> int | None:
        if player_1_choice == Color.RANDOM and player_2_choice == Color.RANDOM:
            return choice([self.player_1_id, self.player_2_id])
        if player_1_choice == Color.RANDOM:
            player_1_choice = player_2_choice
        if player_2_choice == Color.RANDOM:
            player_2_choice = player_1_choice
        if player_1_choice == Color.RED and player_2_choice == Color.YELLOW:
            return self.player_1_id
        if player_1_choice == Color.YELLOW and player_2_choice == Color.RED:
            return self.player_2_id
        return choice([self.player_1_id, self.player_2_id])