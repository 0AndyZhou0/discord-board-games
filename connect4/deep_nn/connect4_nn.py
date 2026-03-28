import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("cogs.connect4.nn")

class Connect4NN(nn.Module):
    def __init__(self, num_channels:int = 256, rows:int = 6, cols:int = 7) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.rows, self.cols = rows, cols
        self.conv1 = nn.Conv2d(1, self.num_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(self.num_channels, self.num_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.bn2 = nn.BatchNorm2d(self.num_channels)
        self.bn3 = nn.BatchNorm2d(self.num_channels)

        self.fc1 = nn.Linear(self.num_channels * self.rows * self.cols, self.num_channels * self.rows * self.cols)
        self.fc2 = nn.Linear(self.num_channels * self.rows * self.cols, self.num_channels * self.rows * self.cols)
        self.fc_bn1 = nn.BatchNorm1d(self.num_channels * self.rows * self.cols)
        self.fc_bn2 = nn.BatchNorm1d(self.num_channels * self.rows * self.cols)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)

        self.policy = nn.Linear(self.num_channels * self.rows * self.cols, self.cols)
        self.evaluation = nn.Linear(self.num_channels * self.rows * self.cols, 1)

    def forward(self, input: torch.Tensor):  # noqa: ANN001, ANN201
        input = input.view(-1, 1, 6, 7)
        input = self.conv1(input)
        input = self.bn1(input)
        input = nn.functional.relu(input)
        # input = torch.square(torch.clamp(torch.relu(input), 0, 1.0)) # Maybe use squared clipped relu
        input = torch.clamp(input, 0, 1.0)
        input = self.conv2(input)
        input = self.bn2(input)
        input = nn.functional.relu(input)
        input = torch.clamp(input, 0, 1.0)
        input = self.conv3(input)
        input = self.bn3(input)
        input = nn.functional.relu(input)
        input = torch.clamp(input, 0, 1.0)
        input = input.view(-1, self.num_channels * self.rows * self.cols)

        # Dropout
        input = self.fc1(input)
        input = self.fc_bn1(input)
        input = nn.functional.relu(input)
        input = torch.clamp(input, 0, 1.0)
        input = self.dropout1(input)

        input = self.fc2(input)
        input = self.fc_bn2(input)
        input = nn.functional.relu(input)
        input = torch.clamp(input, 0, 1.0)
        input = self.dropout2(input)

        policy: torch.Tensor = self.policy(input)
        evaluation: torch.Tensor = self.evaluation(input)
        return torch.log_softmax(policy, dim=1), torch.tanh(evaluation).view(-1)
    
class Connect4NNWrapper:
    def __init__(self, batch_size: int = 64, rows: int = 6, cols: int = 7) -> None:
        self.nn = Connect4NN(num_channels=batch_size, rows=rows, cols=cols)
        self.batch_size = batch_size
        self.rows, self.cols = rows, cols
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            self.nn.cuda()

    def load_model(self, path: str = "./model.pt") -> None:
        self.nn.load_state_dict(torch.load(path, map_location=self.device))

    def save_model(self, path: str = "./model.pt") -> None:
        torch.save(self.nn.state_dict(), path)

    def evaluate_board(self, board: np.array) -> tuple[torch.Tensor, torch.Tensor]:
        board = torch.FloatTensor(board)
        if self.device.type == "cuda":
            board = board.contiguous().cuda()
        board = board.view(1, self.rows, self.cols)
        self.nn.eval()
        with torch.no_grad():
            policy, evaluation = self.nn(board.detach().clone())
            return torch.exp(policy).cpu().numpy()[0], evaluation.data.cpu().numpy()[0]
        
    def train(self, train_set: list[tuple[torch.Tensor, torch.Tensor]], epochs: int = 10, batch_size: int = 64, optimizer: torch.optim.Optimizer = None) -> None:
        if optimizer is None:
            optimizer = torch.optim.Adam(self.nn.parameters())
        
        for epoch in range(epochs):
            print(f"Epoch: {epoch+1}")
            self.nn.train()
            
            policy_losses = []
            evaluation_losses = []
            batch_count = len(train_set) // batch_size # A bit arbitrary
            for batch in range(batch_count):
                samples = np.random.default_rng().choice(len(train_set), size=batch_size)

                boards, train_prob, train_eval = zip(*[train_set[i] for i in samples])
                boards = torch.FloatTensor(np.array(boards))
                train_prob = torch.FloatTensor(np.array(train_prob))
                train_eval = torch.FloatTensor(np.array(train_eval))

                if self.device.type == "cuda":
                    boards = boards.contiguous().cuda()
                    train_prob = train_prob.contiguous().cuda()
                    train_eval = train_eval.contiguous().cuda()

                policy, evaluation = self.nn(boards)

                policy_loss = self.loss_policy(train_prob, policy)
                evaluation_loss = self.loss_evaluation(train_eval, evaluation)
                total_loss = policy_loss + evaluation_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                policy_losses.append(policy_loss.item())
                evaluation_losses.append(evaluation_loss.item())
            
            print(f"Average e ^ policy loss: {sum(policy_losses) / len(policy_losses)}")
            print(f"Average evaluation loss: {sum(evaluation_losses) / len(evaluation_losses)}")

    def loss_policy(self, targets, outputs):  # noqa: ANN001, ANN201
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_evaluation(self, targets, outputs):  # noqa: ANN001, ANN201
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]
    