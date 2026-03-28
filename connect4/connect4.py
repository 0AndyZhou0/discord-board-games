import logging

from discord.ext import commands

from .connect4_accept_view import Connect4AcceptView
from .connect4_bot_choose_color_view import Connect4BotChooseColorView
from .connect4_bot_view import BotMode
from .connect4_view import Connect4View

logger = logging.getLogger("cogs.connect4")


class Connect4(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.command(name="connect4", aliases=["c4"], description="Starts a connect4 game", help="Starts a connect4 game")
    async def connect4(self, ctx: commands.Context, mode: str = commands.parameter(default="multi", description="", displayed_name="mode:")) -> None:
        logger.info("connect4 game started")
        user_id = ctx.author.id
        match mode:
            case "multi":
                await ctx.send(view=Connect4AcceptView(user_id))
            case "solo":
                await ctx.send(view=Connect4View(user_id, user_id))
            case "bot" | "minimax":
                raise NotImplementedError
            case "random":
                await ctx.send(view=Connect4BotChooseColorView(user_id, BotMode.RANDOM))
            case "mcts" | "nn":
                await ctx.send(view=Connect4BotChooseColorView(user_id, BotMode.MCTS_NN))
            case _:
                raise Exception("Not a valid mode")

    @commands.Cog.listener()
    async def on_ready(self) -> None: 
        print("connect4 cog loaded")
