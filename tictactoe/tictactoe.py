from discord.ext import commands

from .tictactoe_accept_view import TicTacToeAcceptView


class TicTacToe(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.command(name="tictactoe", aliases=["ttt"])
    async def tictactoe(self, ctx: commands.Context) -> None:
        print("tictactoe game started")
        await ctx.send("TicTacToe Game Started: ", view=TicTacToeAcceptView(ctx.author.id))

    @commands.Cog.listener()
    async def on_ready(self) -> None: 
        print("tictactoe cog loaded")
