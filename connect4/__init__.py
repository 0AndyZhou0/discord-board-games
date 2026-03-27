from .connect4 import Connect4


async def setup(bot) -> None:  # noqa: ANN001
    await bot.add_cog(Connect4(bot))