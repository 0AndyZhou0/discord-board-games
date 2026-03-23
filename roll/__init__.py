from .roll import Roll


async def setup(bot) -> None:  # noqa: ANN001
    await bot.add_cog(Roll(bot))
