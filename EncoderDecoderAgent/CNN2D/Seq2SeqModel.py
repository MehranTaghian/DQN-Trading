import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        # c is the context
        c = self.encoder(x)
        print(c)
        print(c.squeeze())
        print(c.shape)
        print(c.squeeze().shape)
        output = self.decoder(c.squeeze())
        return output
