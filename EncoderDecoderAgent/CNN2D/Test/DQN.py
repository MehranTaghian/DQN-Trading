import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, state_size, num_classes, action_length=3):
        """
        :param hidden_size: size of the hidden output from attention layer
        :param action_length: Buy, Sell, Idle
        """
        super(DQN, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.Conv1d(state_size, 3, 3, padding=1),
            nn.Conv1d(3, 1, 3, padding=1)
        )

        self.policy_network = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, action_length))

    def forward(self, x):
        # We have to convolve over time axis, thus we change the column of time and price
        x = x.permute(1, 2, 0)
        x = self.conv_encoder(x).squeeze()
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        return self.policy_network(x)
