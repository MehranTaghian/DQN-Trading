import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        # c is the context
        c = self.encoder(x)
        output = self.decoder(c)
        return output
