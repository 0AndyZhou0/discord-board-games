import numpy as np
import torch

from .connect4_nnue import Connect4NNUE


class Connect4NNUEWrapper:
    def __init__(self, batch_size: int = 128, rows: int = 6, cols: int = 7) -> None:
        self.nn = Connect4NNUE()
        self.batch_size = batch_size
        self.rows, self.cols = rows, cols
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            self.nn.cuda()

    def load_model(self, path: str = "./model.pt") -> None:
        self.nn.load_state_dict(torch.load(path, map_location=self.device))

    def save_model(self, path: str = "./model.pt") -> None:
        torch.save(self.nn.state_dict(), path)

    def evaluate_board(self, red_bitboard: np.array, yellow_bitboard: np.array, player: int) -> tuple[torch.Tensor, torch.Tensor]:
        red_bitboard = torch.tensor(red_bitboard, dtype=torch.long).unsqueeze(0)
        yellow_bitboard = torch.tensor(yellow_bitboard, dtype=torch.long).unsqueeze(0)
        if self.device.type == "cuda":
            red_bitboard = red_bitboard.contiguous().cuda()
            yellow_bitboard = yellow_bitboard.contiguous().cuda()
            player = torch.tensor(player, dtype=torch.long).unsqueeze(0).cuda()
        else:
            player = torch.tensor(player, dtype=torch.long).unsqueeze(0)
        self.nn.eval()
        with torch.no_grad():
            evaluation = self.nn(red_bitboard, yellow_bitboard, player)
            return evaluation.data.cpu().numpy()[0]
        
    def accumulator_add(self, row: int, col: int, player: int) -> None:
        self.nn.accumulator_add(row, col, player)

    def accumulator_remove(self, row: int, col: int, player: int) -> None:
        self.nn.accumulator_remove(row, col, player)
    
    def accumulator_forward(self, player: int) -> torch.Tensor:
        self.nn.eval()
        with torch.no_grad():
            # result = self.nn.accumulator_forward(player).data.cpu().numpy()[0]
            # print(result)
            return self.nn.accumulator_forward(player).data.cpu().numpy()[0]

    def train(self, train_set: tuple[torch.LongTensor], epochs: int = 10, batch_size: int = 2048, batch_count: int = 10, optimizer: torch.optim.Optimizer = None, criterion: torch.nn.Module = None) -> None:
        if optimizer is None:
            optimizer = torch.optim.Adam(self.nn.parameters())
        if criterion is None:
            criterion = torch.nn.MSELoss()

        all_red_bitboards, all_yellow_bitboards, all_players, all_train_eval = train_set
        size = len(all_red_bitboards)

        for epoch in range(epochs):
            print(f"Epoch: {epoch+1}")
            self.nn.train()
            
            evaluation_losses = []
            for batch in range(batch_count):
                samples = np.random.default_rng().choice(size, size=batch_size)

                red_bitboards = all_red_bitboards[samples]
                yellow_bitboards = all_yellow_bitboards[samples]
                players = all_players[samples]
                train_eval = all_train_eval[samples].float()

                if self.device.type == "cuda":
                    red_bitboards = red_bitboards.contiguous().cuda()
                    yellow_bitboards = yellow_bitboards.contiguous().cuda()
                    players = players.contiguous().cuda()
                    train_eval = train_eval.contiguous().cuda()
                
                evaluation = self.nn(red_bitboards, yellow_bitboards, players)

                evaluation_loss = criterion(train_eval.view(-1), evaluation)

                optimizer.zero_grad()
                evaluation_loss.backward()
                optimizer.step()

                evaluation_losses.append(evaluation_loss.item())
            
            print(f"Average evaluation loss: {sum(evaluation_losses) / len(evaluation_losses)}")
