import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, attention, decoder):
        super().__init__()
        self.encoder = encoder
        # self.attention = attention
        self.decoder = decoder

    def forward(self, x):
        output, hidden = self.encoder(x)
        # att_output = self.attention(output, hidden)
        # final_output = self.decoder(att_output)
        final_output = self.decoder(hidden)

        # return final_output, hidden
        return final_output
