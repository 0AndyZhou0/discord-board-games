import logging
from enum import IntEnum
from pathlib import Path
from random import choice

import discord
import numpy as np

# from .connect4_game import Color, Connect4Game
from .deep_nn.connect4 import Color, Connect4
from .deep_nn.connect4_mcts import Connect4MCTS
from .deep_nn.connect4_nn import Connect4NNWrapper

from .nnue.connect4_translator import Connect4Translator

logger = logging.getLogger("cogs.connect4")


class BotMode(IntEnum):
    RANDOM = 0
    MCTS_NN = 1
    MINIMAX = 2

class Connect4Button(discord.ui.Button["Connect4BotView"]):
    def __init__(self, col: int) -> None:
        super().__init__(style=discord.ButtonStyle.primary, label=str(col+1), custom_id=str(col), row=1)
        self.col = col

    async def callback(self, interaction: discord.Interaction) -> None:
        view = self.view
        col = self.col

        user_id = interaction.user.id
        if user_id != view.player_id:
            await interaction.response.send_message(content="You are not in the game", ephemeral=True)
            return
        
        # Player Move
        move = view.drop_piece(col)
        if view.is_column_full(col):
            self.disabled = True
        winner = view.check_for_win(*move)
        if winner is not None:
            if winner == 0:
                view.text_display.content = f"<@{view.player_id}> and <@{view.bot_id}> tied!\n{view.emoji_board}"
            else:
                view.text_display.content = f"<@{view.player_id}> wins against the bot!\n{view.emoji_board}"
            view.stop_game()
            view.stop()
            await interaction.response.edit_message(view=view)
            return
        
        # Bot Move
        move = view.bot_move()
        if view.is_column_full(move[1]):
            for button in view.action_row.children + view.action_row2.children:
                if button.col == move[1]:
                    button.disabled = True
        winner = view.check_for_win(*move)
        if winner is not None:
            if winner == 0:
                view.text_display.content = f"<@{view.player_id}> and <@{view.bot_id}> tied!\n{view.emoji_board}"
            else:
                view.text_display.content = f"The bot wins against <@{view.player_id}>!\n{view.emoji_board}"
            view.stop_game()
            view.stop()
            await interaction.response.edit_message(view=view)
            return
        
        # Update Board
        await interaction.response.edit_message(view=view)
        
        

class Connect4BotView(discord.ui.LayoutView):
    def __init__(self, player_id: int, player_color: Color, bot_mode: BotMode) -> None:
        super().__init__()
        logger.info("Creating Connect4 Game")
        self.current_player = Color.RED
        self.player_id = player_id
        self.player_color = player_color
        self.bot_mode = bot_mode
        self.moves = ""

        if bot_mode == BotMode.MCTS_NN:
            path = Path(__file__).parent / "deep_nn" / "models"
            nn = Connect4NNWrapper(num_channels=32)
            nn.load_model(path / f"best{nn.num_channels}.pt")
            self.mcts = Connect4MCTS(nn)

        if bot_mode == BotMode.MINIMAX:
            self.translator = Connect4Translator()

        # Create board
        self.board: np.array = Connect4.get_empty_board()
        self.emoji_board: str = Connect4.get_emoji_board(self.board)

        # Create text display
        self.text_display = discord.ui.TextDisplay(f"It is <@{self.player_id}>'s turn\n{self.emoji_board}")
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

        # Bot first move
        if self.player_color == Color.YELLOW:
            self.bot_move()

    
    def drop_piece(self, col: int) -> tuple[int, int]:
        match self.current_player:
            case Color.RED:
                move, next_player = Connect4.drop_piece(self.board, col, self.current_player)
                self.current_player = Color.YELLOW
            case Color.YELLOW:
                move, next_player = Connect4.drop_piece(self.board, col, self.current_player)
                self.current_player = Color.RED
            case _:
                raise Exception("Invalid Player")
        self.emoji_board = Connect4.get_emoji_board(self.board)
        self.text_display.content = f"It is <@{self.player_id}>'s turn\n{self.emoji_board}"
        self.moves += str(col + 1)
        return move
    
    def is_column_full(self, col: int) -> bool:
        return Connect4.is_column_full(self.board, col)
    
    def check_for_win(self, row: int, col: int) -> Color | None:
        return Connect4.get_game_win(self.board, row, col)
    
    def stop_game(self) -> None:
        for button in self.action_row.children:
            button.disabled = True
        for button in self.action_row2.children:
            button.disabled = True

    def bot_move(self) -> tuple[int, int]:
        match self.bot_mode:
            case BotMode.RANDOM:
                bot_col = self.random_move()
            case BotMode.MINIMAX:
                bot_col = self.minimax_move()
            case BotMode.MCTS_NN:
                bot_col = self.mcts_nn_move()
            case _:
                raise Exception("Invalid bot mode")
        return self.drop_piece(bot_col)
            
    def random_move(self) -> int:
        valid_cols = Connect4.get_valid_cols(self.board)
        return choice(valid_cols)

    def mcts_nn_move(self) -> int:
        canonical_board = Connect4.get_canonical_board(self.board, self.current_player)
        return np.argmax(self.mcts.get_best_actions(canonical_board, 20))
    
    def minimax_move(self) -> int:
        # return self.translator.get_best_col_from_board(self.board, self.current_player, self.moves)
        return self.translator.get_best_col_from_board(self.board, self.current_player)