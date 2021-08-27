import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        """
        :param input_size: 5 which is OHLC + trend
        """
        super(EncoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)
        # self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, x):
        """
        :param x: if the input x is a batch, its size is of the form [window_size, batch_size, input_size]
        thus, the output of GRU would be of shape [window_size, batch_size, hidden_size].
        e.g. output[:, 0, :] is the output sequence of the first element in the batch.
        The hidden is of the shape [1, batch_size, hidden_size]
        """

        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        hidden = self.initHidden(x.shape[1])

        output, hidden = self.gru(x, hidden)
        # output, hidden = self.gru(x)

        # cell = self.initHidden(x.shape[1])
        # output, (hidden, cell) = self.lstm(x, (hidden, cell))

        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)
