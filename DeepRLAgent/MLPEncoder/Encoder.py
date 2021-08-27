import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, num_classes, state_size):
        """

        :param state_size: we give OHLC as input to the network
        :param action_length: Buy, Sell, Idle
        """
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.BatchNorm1d(128),
            # nn.Linear(128, 256),
            # nn.BatchNorm1d(256),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
