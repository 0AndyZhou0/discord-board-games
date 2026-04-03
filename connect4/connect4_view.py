import logging

import discord
import numpy as np

from .connect4_game import Color, Connect4Game

logger = logging.getLogger("cogs.connect4")

class Connect4Button(discord.ui.Button["Connect4View"]):
    def __init__(self, col: int) -> None:
        super().__init__(style=discord.ButtonStyle.primary, label=str(col+1), custom_id=str(col), row=1)
        self.col = col

    async def callback(self, interaction: discord.Interaction) -> None:
        view = self.view
        col = self.col

        user_id = interaction.user.id
        if user_id not in (view.player_red, view.player_yellow):
            await interaction.response.send_message(content="You are not in the game", ephemeral=True)
            return
        if user_id != view.get_player_id(view.current_player):
            await interaction.response.send_message(content="It is not your turn", ephemeral=True)
            return
        
        move = view.drop_piece(col)
        if view.is_column_full(col):
            self.disabled = True
        winner = view.check_for_win(*move)
        if winner is not None:
            if view.is_game_tie():
                view.text_display.content = f"<@{view.player_red}> and <@{view.player_yellow}> tied!\n{view.emoji_board}"
            else:
                view.text_display.content = f"<@{view.get_player_id(winner)}> wins against <@{view.get_player_id(-winner)}>\n{view.emoji_board}"
            view.stop_game()
            view.stop()
        await interaction.response.edit_message(view=view)

class Connect4View(discord.ui.LayoutView):
    def __init__(self, player_red: int, player_yellow: int) -> None:
        super().__init__()
        logger.info("Creating Connect4 Game")
        self.current_player = Color.RED
        self.player_red = player_red
        self.player_yellow = player_yellow

        # Create board
        self.board: np.array = Connect4Game.get_empty_board()
        self.emoji_board: str = Connect4Game.get_emoji_board(self.board)

        # Create text display
        self.text_display = discord.ui.TextDisplay(f"It is <@{self.player_red}>'s turn\n{self.emoji_board}")
        self.add_item(self.text_display)

        # Create buttons
        self.action_row = discord.ui.ActionRow()
        for c in range(4): # Max num of buttons per action row is 5
            self.action_row.add_item(Connect4Button(c))
        self.action_row2 = discord.ui.ActionRow()
        for c in range(4, 7): 
            self.action_row2.add_item(Connect4Button(c))
        self.add_item(self.action_row)
        self.add_item(self.action_row2)
    
    def drop_piece(self, col: int) -> tuple[int, int]:
        match self.current_player:
            case Color.RED:
                move, next_player = Connect4Game.drop_piece(self.board, col, self.current_player)
                self.current_player = Color.YELLOW
            case Color.YELLOW:
                move, next_player = Connect4Game.drop_piece(self.board, col, self.current_player)
                self.current_player = Color.RED
            case _:
                raise Exception("Invalid Player")
        self.emoji_board = Connect4Game.get_emoji_board(self.board)
        self.text_display.content = f"It is <@{self.get_player_id(self.current_player)}>'s turn\n{self.emoji_board}"
        return move
    
    def is_column_full(self, col: int) -> bool:
        return Connect4Game.is_column_full(self.board, col)
    
    def is_game_tie(self) -> bool:
        return Connect4Game.is_game_tie(self.board)
    
    def check_for_win(self, row: int, col: int) -> Color | None:
        return Connect4Game.get_game_win(self.board, row, col)
    
    def get_player_id(self, color: Color) -> int:
        match color:
            case Color.RED:
                return self.player_red
            case Color.YELLOW:
                return self.player_yellow
            case _:
                raise Exception("Invalid Color")
    
    def stop_game(self) -> None:
        for button in self.action_row.children:
            button.disabled = True
        for button in self.action_row2.children:
            button.disabled = True