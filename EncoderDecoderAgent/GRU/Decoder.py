import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, hidden_size, action_length=3):
        """
        :param hidden_size: size of the hidden output from attention layer
        :param action_length: Buy, Sell, Idle
        """
        super(Decoder, self).__init__()

        self.policy_network = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, action_length))

        # self.layer1 = nn.Linear(hidden_size, 128)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.layer2 = nn.Linear(128, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.out = nn.Linear(256, action_length)

    def forward(self, x):
        # if x.shape[0] > 1:
        #     x = F.relu(self.bn1(self.layer1(x)))
        #     x = F.relu(self.bn2(self.layer2(x)))
        # else:
        #     x = F.relu(self.layer1(x))
        #     x = F.relu(self.layer2(x))
        # return self.out(x)

        x = x.squeeze().unsqueeze(0) if len(x.squeeze().shape) < 2 else x.squeeze()
        output = self.policy_network(x).squeeze()
        return output if len(output.shape) > 1 else output.unsqueeze(0)
