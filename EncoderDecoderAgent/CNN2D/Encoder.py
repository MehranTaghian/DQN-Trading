import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, num_classes, state_size, window_size):
        """

        :param state_size: we give OHLC as input to the network
        :param action_length: Buy, Sell, Idle
        """
        super(Encoder, self).__init__()
        self.encoder = nn.Conv2d(1, num_classes, (window_size, state_size), 1)

    def forward(self, x):
        x = x.permute(1, 0, 2).unsqueeze(1)
        x = self.encoder(x)
        x = x.squeeze() if len(x.squeeze().shape) > 1 else x.squeeze().unsqueeze(0)
        return x
