import torch 

from torch import nn
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, latent_size, out_size, hidden_size=16):
        super(Generator, self).__init__()

        self.latent_size = latent_size

        # One modification from the original, I'm not sure why
        # the authors used 2d convolutions on vectors. Doesn't
        # really make a lot of sense to me... so I'm just using 
        # linear layers to see if it still works
        self.linears = nn.Sequential(
            nn.Linear(latent_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
            nn.Sigmoid()
        )

        self.recurrent = nn.LSTM(out_size, out_size)

    def forward(self, batch_size, sentence_len=5):
        x = Variable(torch.empty(batch_size, self.latent_size).normal_())
        
        x = self.linears(x)
        h_n, c_n = self.recurrent(x)
        words = [h_n]

        for _ in range(sentence_len):
            h_n, c_n = self.recurrent(h_n, c_n) 
            words.append(h_n)

        return torch.cat(words, dim=0)


class Discriminator(nn.Module):
    def __init__(self, embed_size, hidden_size=16):
        super(Discriminator, self).__init__()

        self.embed_size = embed_size

        self.recurrent = nn.Sequential(
            nn.LSTM(embed_size, hidden_size, num_layers=3), 
            nn.Sigmoid()
        )

        self.decider = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x, _ = self.recurrent(x)
        x = x.squeeze(0)

        return self.decider(x)

