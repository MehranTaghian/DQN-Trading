import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, window_size, device):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.device = device

        self.attn = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, encoder_output, encoder_hidden):
        """
        :param encoder_output: shape is [max_length, 1, hidden_size]
        :param encoder_hidden: shape is [1, 1, hidden_size]
        :return:
        """
        # shape is [max_length, 1, hidden_size]
        # encoder_hidden.shape[1] is the batch_size
        hidden_temp = torch.zeros(self.window_size, encoder_hidden.shape[1], self.hidden_size, device=self.device)

        hidden_temp[torch.arange(self.window_size)] = encoder_hidden[0]

        # shape is [max_length, hidden_size * 2]
        att_input = torch.cat((encoder_output, hidden_temp), dim=2)

        # shape is [max_length, 1]
        att_weights = nn.functional.softmax(self.attn(att_input), dim=0)

        # shape is [1, hidden_size] and this is the state vector fed to the policy network
        att_applied = torch.bmm(att_weights.permute(1, 2, 0), encoder_output.transpose(0, 1))

        return att_applied
