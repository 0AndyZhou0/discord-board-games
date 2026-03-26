import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("cogs.tictactoe.nn.nn")

class TicTacToeNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 3 * 3, 64 * 3 * 3)
        self.fc2 = nn.Linear(64 * 3 * 3, 64 * 3 * 3)
        self.fc_bn1 = nn.BatchNorm1d(64 * 3 * 3)
        self.fc_bn2 = nn.BatchNorm1d(64 * 3 * 3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)

        self.policy = nn.Linear(64 * 3 * 3, 9)
        self.evaluation = nn.Linear(64 * 3 * 3, 1)

    def forward(self, input: torch.Tensor):  # noqa: ANN001, ANN201
        input = input.view(-1, 1, 3, 3)
        input = self.conv1(input)
        input = self.bn1(input)
        input = nn.functional.relu(input)
        input = self.conv2(input)
        input = self.bn2(input)
        input = nn.functional.relu(input)
        input = self.conv3(input)
        input = self.bn3(input)
        input = nn.functional.relu(input)
        input = input.view(-1, 64 * 3 * 3)

        # Dropout
        input = self.fc1(input)
        input = self.fc_bn1(input)
        input = nn.functional.relu(input)
        input = self.dropout1(input)

        input = self.fc2(input)
        input = self.fc_bn2(input)
        input = nn.functional.relu(input)
        input = self.dropout2(input)

        policy: torch.Tensor = self.policy(input)
        evaluation: torch.Tensor = self.evaluation(input)
        return torch.log_softmax(policy, dim=1), torch.tanh(evaluation).view(-1)
    
class TicTacToeNNWrapper:
    def __init__(self, nn: TicTacToeNN, device: torch.device) -> None:
        self.nn = nn
        self.device = device

        if device.type == "cuda":
            self.nn.cuda()

    def load_model(self, path: str) -> None:
        self.nn.load_state_dict(torch.load(path, map_location=self.device))

    def save_model(self, path: str) -> None:
        torch.save(self.nn.state_dict(), path)

    def evaluate_board(self, board: np.array) -> tuple[torch.Tensor, torch.Tensor]:
        board = torch.FloatTensor(board)
        if self.device.type == "cuda":
            board = board.contiguous().cuda()
        board = board.view(1, 3, 3)
        self.nn.eval()
        with torch.no_grad():
            # policy, evaluation = self.nn(torch.tensor(board, device=self.device))
            policy, evaluation = self.nn(board.detach().clone())
            return torch.exp(policy).cpu().numpy()[0], evaluation.data.cpu().numpy()[0]
        
    def train(self, train_set: list[tuple[torch.Tensor, torch.Tensor]], epochs: int, batch_size: int, optimizer: torch.optim.Optimizer = None) -> None:
        if optimizer is None:
            optimizer = torch.optim.Adam(self.nn.parameters())
        
        for epoch in range(epochs):
            print(f"Epoch: {epoch+1}")
            self.nn.train()
            
            policy_losses = []
            evaluation_losses = []
            batch_count = len(train_set) // batch_size
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