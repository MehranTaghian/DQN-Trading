import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, input_size, action_length=3):
        """

        :param state_length: we give OHLC as input to the network
        :param action_length: Buy, Sell, Idle
        """
        super(Decoder, self).__init__()

        self.policy_network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, action_length))

    def forward(self, x):
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        return self.policy_network(x)
