import random

from discord.ext import commands


class Roll(commands.Cog):
    def __init__(self, bot) -> None:  # noqa: ANN001
        self.bot = bot

    @commands.command()
    async def roll(self, ctx) -> None:  # noqa: ANN001
        message = ctx.message

        message_content = message.content
        message_content = message_content.replace(self.bot.command_prefix + "roll", " ")

        try:
            max_roll = int(message_content)
        except ValueError:
            max_roll = 100

        if max_roll > 1e108:
            await message.channel.send("max roll is somewhere around 1e108 because doubles")
            return
        await message.channel.send(f"<@{message.author.id}> rolled a {str(round(random.uniform(1, max_roll)))}")

    @commands.Cog.listener()
    async def on_ready(self) -> None: 
        print("roll cog loaded")