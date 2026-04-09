import logging
from random import choice

import discord

from .connect4_bot_view import BotMode, Connect4BotView
from .connect4_game import Color

logger = logging.getLogger("cogs.connect4")


class ChooseColorButton(discord.ui.Button['Connect4BotChooseColorView']):
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

        if user_id != view.player_1_id:
            interaction.response.send_message(content="You are not in the game", ephemeral=True)
            return
        
        if self.symbol == Color.RANDOM:
            player_1_choice = choice([Color.RED, Color.YELLOW])
        else:
            player_1_choice = self.symbol
        
        match player_1_choice:
            case Color.RED:
                logger.info(f"<@{view.player_1_id}> chose RED. <@{view.player_1_id}> goes first")
                await interaction.response.defer()
                await interaction.followup.edit_message(message_id=interaction.message.id, view=Connect4BotView(view.player_1_id, Color.RED, view.bot_mode))
                return
            case Color.YELLOW:
                logger.info(f"<@{view.player_1_id}> chose YELLOW. Bot goes first")
                await interaction.response.defer()
                await interaction.followup.edit_message(message_id=interaction.message.id, view=Connect4BotView(view.player_1_id, Color.YELLOW, view.bot_mode))
                return

        raise Exception("Impossible to reach code while choosing color")


class Connect4BotChooseColorView(discord.ui.LayoutView):
    def __init__(self, player_1_id: int, bot_mode: BotMode) -> None:
        super().__init__()
        self.player_1_id = player_1_id
        self.bot_mode = bot_mode

        action_row = discord.ui.ActionRow()
        action_row.add_item(ChooseColorButton(Color.RED))
        action_row.add_item(ChooseColorButton(Color.YELLOW))
        action_row.add_item(ChooseColorButton(Color.RANDOM))
        self.add_item(action_row)
    
    def determine_red_player(self, player_1_choice: Color) -> int | None:
        if player_1_choice == Color.RED:
            return self.player_1_id
        if player_1_choice == Color.YELLOW:
            return -1
        return choice([self.player_1_id, -1])