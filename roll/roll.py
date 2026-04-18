import random

from discord.ext import commands


class Roll(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:  # noqa: ANN001
        self.bot = bot

    @commands.command()
    @commands.cooldown(3, 5, commands.BucketType.user)
    async def roll(self, ctx: commands.Context) -> None:  # noqa: ANN001
        message = ctx.message

        message_content = message.content
        message_content = message_content.replace(self.bot.command_prefix + "roll", " ")

        try:
            max_roll = int(message_content)
        except ValueError:
            max_roll = 100
        
        print("Interaction:", ctx.interaction)

        if max_roll > 1e108:
            await ctx.reply("max roll is somewhere around 1e108 because doubles")
            return
        await ctx.send(f"<@{message.author.id}> rolled a {str(round(random.uniform(1, max_roll)))}")

    @commands.Cog.listener()
    async def on_ready(self) -> None: 
        print("roll cog loaded")

    @roll.error
    async def on_error(self, ctx: commands.Context, error: Exception) -> None:
        # print("Interaction:", ctx.interaction)
        if isinstance(error, commands.CommandOnCooldown):
            await ctx.reply(f"Please wait {round(error.retry_after, 2)} seconds before using this command again.", ephemeral=True, delete_after=error.retry_after)