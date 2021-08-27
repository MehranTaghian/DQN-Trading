import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, window_size, output_size):
        super(AttentionLayer, self).__init__()
        self.attn = nn.Linear(window_size, output_size)

    def forward(self, x):
        """
        :param x: output of conv layer with dimension [batch_size, window_size, OHLC]
        :return:
        """
        x = x.squeeze() if len(x.squeeze().shape) > 1 else x.squeeze().unsqueeze(0)
        return F.softmax(self.attn(x), dim=1)
