import torch.nn as nn
from .Encoder import Encoder
from .Decoder import Decoder


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        # c is the context
        c = self.encoder(x)
        output = self.decoder(c.squeeze())
        return output
