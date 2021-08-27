import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, state_size):
        """

        :param state_size: we give OHLC as input to the network
        """
        super(Encoder, self).__init__()
        self.conv_encoder = nn.Sequential(
            nn.Conv1d(state_size, 2, 3, padding=1),
            nn.Conv1d(2, 1, 3, padding=1)
        )

    def forward(self, x):
        # We have to convolve over time axis, thus we change the column of time and price
        x = x.permute(1, 2, 0)
        x = self.conv_encoder(x)
        return x
