from discord.ext import commands

from .tictactoe_accept_view import TicTacToeAcceptView
from .tictactoe_bot_choose_symbol_view import TicTacToeBotChooseSymbolView
from .tictactoe_view import TicTacToeView


class TicTacToe(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @commands.command(name="tictactoe", aliases=["ttt"], description="Starts a tictactoe game", help="Starts a tictactoe game")
    async def tictactoe(self, ctx: commands.Context, mode: str = commands.parameter(default=None, description="Optional[solo, bot]", displayed_name="mode:")) -> None:
        print("tictactoe game started")
        user = ctx.author.id
        match mode:
            case "solo":
                await ctx.send(content=f"It is now <@{user}>'s turn", view=TicTacToeView(user, user))
            case "bot":
                await ctx.send(content="Choose Symbol You Want To Play", view=TicTacToeBotChooseSymbolView(user))
            case _:
                await ctx.send("TicTacToe Game Started: ", view=TicTacToeAcceptView(user))

    @commands.Cog.listener()
    async def on_ready(self) -> None: 
        print("tictactoe cog loaded")
