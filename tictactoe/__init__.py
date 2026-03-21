from .tictactoe import TicTacToe


async def setup(bot) -> None:  # noqa: ANN001
    await bot.add_cog(TicTacToe(bot))
