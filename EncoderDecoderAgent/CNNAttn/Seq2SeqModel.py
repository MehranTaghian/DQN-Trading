import torch.nn as nn
from .Encoder import Encoder
from .Decoder import Decoder


class Seq2Seq(nn.Module):
    def __init__(self, encoder, attention, decoder):
        super().__init__()
        self.encoder = encoder
        self.attn = attention
        self.decoder = decoder

    def forward(self, x):
        # c is the context
        encoder_out = self.encoder(x)
        att_output = self.attn(encoder_out)
        output = self.decoder(att_output)
        return output
