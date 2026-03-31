import logging

import torch
import torch.nn as nn

from .connect4_color import Color

logger = logging.getLogger("cogs.connect4.nn")

class CReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.clamp(input, min=0, max=1)

class Connect4NNUE(nn.Module):
    """[-1,1] eval 1 means yellow win, -1 means red win, 0 means draw"""
    def __init__(self, rows:int = 6, cols:int = 7, accumulator_size:int = 256) -> None:
        super().__init__()
        self.rows, self.cols = rows, cols
        self.red_accumulator = nn.Linear(self.rows * self.cols * 2, accumulator_size)
        self.yellow_accumulator = nn.Linear(self.rows * self.cols * 2, accumulator_size) # Just switch red_bitboard and yellow_bitboard in the input
        self.pooling1 = nn.Linear(accumulator_size * 2, 8) # Just switch red_accumulator and yellow_accumulator in the input depending on the player
        self.pooling2 = nn.Linear(8, 8)
        self.evaluation = nn.Linear(8, 1)
        self.crelu = CReLU()

    def longs_to_tensor(self, red_bitboard: torch.Tensor, yellow_bitboard: torch.Tensor) -> torch.Tensor:
        mask = 2 ** torch.arange(self.rows * self.cols, dtype=torch.long, device=red_bitboard.device)
        red_tensor = red_bitboard.detach().clone().unsqueeze(-1).bitwise_and(mask).ne(0)
        yellow_tensor = yellow_bitboard.detach().clone().unsqueeze(-1).bitwise_and(mask).ne(0)
        combined = torch.cat((red_tensor, yellow_tensor), dim=-1).view(-1, self.rows * self.cols * 2)
        return combined

    def forward(self, red_bitboard: torch.Tensor, yellow_bitboard: torch.Tensor, player: torch.Tensor) -> torch.Tensor:
        red_input = self.longs_to_tensor(red_bitboard, yellow_bitboard)
        yellow_input = self.longs_to_tensor(yellow_bitboard, red_bitboard)
        red_input = red_input.float()
        yellow_input = yellow_input.float()

        red_input = self.red_accumulator(red_input)
        yellow_input = self.yellow_accumulator(yellow_input)
        self.red_accumulator_features = red_input
        self.yellow_accumulator_features = yellow_input
        # if player == Color.RED:
        #     combined = torch.cat((red_input, yellow_input), dim=-1)
        # else:
        #     combined = torch.cat((yellow_input, red_input), dim=-1)

        batch_size = red_input.shape[0]
        input_order = ((1 - player) // 2).view(batch_size, 1)
        combined = (torch.cat((red_input, yellow_input), dim=-1)*input_order) + (torch.cat((yellow_input, red_input), dim=-1)*(1-input_order))
        combined = self.crelu(combined)

        combined = self.pooling1(combined)
        combined = self.crelu(combined)

        combined = self.pooling2(combined)
        combined = self.crelu(combined)

        eval = self.evaluation(combined)
        self.last_eval = eval
        return -eval.view(-1)  # noqa: RET504
    
    def accumulator_add(self, row: int, col: int, player: Color) -> None:
        if player == Color.RED:
            weight_col = row * self.cols + col
            red_weight_column = self.red_accumulator.weight[:, weight_col]
            self.red_accumulator_features += red_weight_column
            yellow_weight_column = self.yellow_accumulator.weight[:, weight_col + self.rows * self.cols]
            self.yellow_accumulator_features += yellow_weight_column
        else:
            weight_col = row * self.cols + col
            yellow_weight_column = self.yellow_accumulator.weight[:, weight_col]
            self.yellow_accumulator_features += yellow_weight_column
            red_weight_column = self.red_accumulator.weight[:, weight_col + self.rows * self.cols]
            self.red_accumulator_features += red_weight_column

    def accumulator_remove(self, row: int, col: int, player: Color) -> None:
        if player == Color.RED:
            weight_col = row * self.cols + col
            red_weight_column = self.red_accumulator.weight[:, weight_col]
            self.red_accumulator_features -= red_weight_column
            yellow_weight_column = self.yellow_accumulator.weight[:, weight_col + self.rows * self.cols]
            self.yellow_accumulator_features -= yellow_weight_column
        else:
            weight_col = row * self.cols + col
            yellow_weight_column = self.yellow_accumulator.weight[:, weight_col]
            self.yellow_accumulator_features -= yellow_weight_column
            red_weight_column = self.red_accumulator.weight[:, weight_col + self.rows * self.cols]
            self.red_accumulator_features -= red_weight_column
    
    i = 0
    def accumulator_forward(self, player: Color) -> torch.Tensor:
        """
        player is the current player's turn
        """
        red_input = self.red_accumulator_features
        yellow_input = self.yellow_accumulator_features
        if player == Color.RED:
            combined = torch.cat((red_input, yellow_input), dim=-1)
        else:
            combined = torch.cat((yellow_input, red_input), dim=-1)
        combined = self.crelu(combined)

        combined = self.pooling1(combined)
        combined = self.crelu(combined)

        combined = self.pooling2(combined)
        combined = self.crelu(combined)

        eval = self.evaluation(combined)
        self.last_eval = eval
        return -eval.view(-1)  # noqa: RET504
    
    def get_last_evaluation(self) -> torch.Tensor:
        return -self.last_eval
