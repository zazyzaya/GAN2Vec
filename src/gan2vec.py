import torch 

from torch import nn
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, latent_size, out_size, hidden_size=16, 
                max_len=20, min_len=3, num_layers=3):
        super(Generator, self).__init__()

        self.latent_size = latent_size
        self.MAX_LEN = max_len
        self.MIN_LEN = min_len 

        self.recurrent = nn.LSTM( 
            latent_size,
            out_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.tan = nn.Tanh()

    def forward(self, batch_size, sentence_len=5):
        # Tell the encoder how long the sentence will be 
        x = Variable(torch.empty(batch_size, sentence_len, self.latent_size).normal_())
        h_n, (x,c) = self.recurrent(x)

        return self.tan(h_n)


class Discriminator(nn.Module):
    def __init__(self, embed_size, hidden_size=16):
        super(Discriminator, self).__init__()

        self.embed_size = embed_size

        self.recurrent = nn.Sequential(
            nn.LSTM(
                embed_size, 
                hidden_size, 
                num_layers=3, 
                batch_first=True
            ), 
        )

        self.decider = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, (x, _) = self.recurrent(x)
        x = x[-1]
        return self.decider(x)
