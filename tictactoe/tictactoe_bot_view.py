import logging
from enum import IntEnum
from pathlib import Path
from random import choice

import discord
import numpy as np
import torch

from .nn.tictactoe import TicTacToe
from .nn.tictactoe_mcts import TicTacToe_MCTS
from .nn.tictactoe_nn import TicTacToeNN, TicTacToeNNWrapper
from .tictactoe_bot import TicTacToeBot

logger = logging.getLogger("cogs.tictactoe.bot")
logger.setLevel(logging.INFO)

class Bot_Mode(IntEnum):
    RANDOM = 0
    MINIMAX = 1
    MCTS_NN = 2

class TicTacToeButton(discord.ui.Button['TicTacToeBotView']):
    def __init__(self, x: int, y: int) -> None:
        super().__init__(style=discord.ButtonStyle.secondary, label="\u200b", row=x)
        self.x = x
        self.y = y

    async def callback(self, interaction: discord.Interaction) -> None:
        assert self.view is not None
        view: TicTacToeBotView = self.view

        user_id = interaction.user.id

        if user_id != view.player:
            await interaction.response.send_message(content="You are not in the game", ephemeral=True)
            return

        state = view.board[self.x][self.y]
        if state in (Symbol.X, Symbol.O):
            return

        if view.player_symbol == Symbol.X:
            self.style = discord.ButtonStyle.danger
            self.label = 'X'
            self.disabled = True
            view.place_symbol(self.x, self.y, Symbol.X)
        elif view.player_symbol == Symbol.O:
            self.style = discord.ButtonStyle.success
            self.label = 'O'
            self.disabled = True
            view.place_symbol(self.x, self.y, Symbol.O)

        winner = view.winner
        if winner is not None: # TODO: Maybe save board positions and results for training?
            if winner == Symbol.TIE:
                content = f"<@{user_id}> and the bot tied!"
            else:
                content = f"<@{user_id}> won against the bot!"

            view.stop_game()
            view.stop()

            await interaction.response.edit_message(content=content, view=view)
            return

        # Bot Move
        view.bot_move()
        content = f"It is now <@{user_id}>'s turn"

        winner = view.winner
        if winner is not None:
            if winner == Symbol.TIE:
                content = f"<@{user_id}> and the bot tied!"
            else:
                content = f"The bot won against <@{user_id}>!"

            view.stop_game()
            view.stop()

            await interaction.response.edit_message(content=content, view=view)
            return

        await interaction.response.edit_message(content=content, view=view)
        return

class Symbol(IntEnum):
    X = -1
    EMPTY = 0
    O = 1
    TIE = 2
    RANDOM = 3

class TicTacToeBotView(discord.ui.View):
    def __init__(self, player_X_id: int, player_O_id: int, mode: Bot_Mode) -> None:
        super().__init__()

        self.player = player_X_id if player_X_id != -1 else player_O_id
        self.player_symbol = Symbol.X if player_X_id != -1 else Symbol.O
        self.bot_symbol = Symbol.X if player_X_id == -1 else Symbol.O
        self.mode = mode
        if mode == Bot_Mode.MCTS_NN:
            path = Path(__file__).parent / "nn" / "models"
            nn_wrapper = TicTacToeNNWrapper(TicTacToeNN(), torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            nn_wrapper.load_model(path / "best.pt")
            self.mcts = TicTacToe_MCTS(nn_wrapper, 1)

        self.board = [[Symbol.EMPTY] * 3 for _ in range(3)]
        self.winner = None
        
        for x in range(3):
            for y in range(3):
                self.add_item(TicTacToeButton(x, y))

        # Bot First Move
        if self.bot_symbol == Symbol.X:
            self.bot_move()
    
    def stop_game(self) -> None:
        for button in self.children:
            button.disabled = True
    
    def place_symbol(self, x: int, y: int, symbol: Symbol) -> None:
        self.board[x][y] = symbol
        self.winner = self.check_for_win()

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
        
    def bot_move(self) -> None:
        match self.mode:
            case Bot_Mode.RANDOM:
                self.random_move()
            case Bot_Mode.MINIMAX:
                self.minimax_move()
            case Bot_Mode.MCTS_NN:
                self.mcts_nn_move()
            case _:
                raise Exception("Invalid bot mode")

    def random_move(self) -> tuple[int, int] | None:
        empty_squares = []
        for x in range(3):
            for y in range(3):
                if self.board[x][y] == Symbol.EMPTY:
                    empty_squares.append((x, y))

        if empty_squares:
            x, y = choice(empty_squares)
            self.place_bot_symbol(x, y)
            return (x, y)

        return None
    
    def minimax_move(self) -> tuple[int, int] | None:
        """
        Will return non losing move or winning move if there is one
        """
        # x, y = TicTacToeBot.find_best_move(self.board, self.bot_symbol)
        x, y = TicTacToeBot.find_best_move_first_weighted(self.board, self.bot_symbol)
        self.place_bot_symbol(x, y)
        return (x, y)
    
    def mcts_nn_move(self) -> tuple[int, int] | None:
        assert self.mcts is not None
        canonical_board = TicTacToe.get_canonical_board(self.board, self.bot_symbol)
        action = np.argmax(self.mcts.get_best_actions(canonical_board, 10))
        x, y = action // 3, action % 3
        self.place_bot_symbol(x, y)
        return (x, y)

    
    def place_bot_symbol(self, x: int, y: int) -> None:
        self.place_symbol(x, y, self.bot_symbol)
        for button in self.children:
            if button.x == x and button.y == y:
                if self.bot_symbol == Symbol.X:
                    button.style = discord.ButtonStyle.danger
                    button.label = 'X'
                    button.disabled = True
                elif self.bot_symbol == Symbol.O:
                    button.style = discord.ButtonStyle.success
                    button.label = 'O'
                    button.disabled = True