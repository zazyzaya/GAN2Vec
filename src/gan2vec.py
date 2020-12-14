import torch 

from torch import nn
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, latent_size, out_size, hidden_size=16, max_len=20, min_len=3):
        super(Generator, self).__init__()

        self.latent_size = latent_size
        self.MAX_LEN = max_len
        self.MIN_LEN = min_len 

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

        self.recurrent = nn.LSTM(out_size, out_size, batch_first=True)

    def forward(self, batch_size, sentence_len=5):
        len_one_hot = torch.zeros(batch_size, self.MAX_LEN-self.MIN_LEN)
        len_one_hot[:, self.MAX_LEN-sentence_len] = 1

        # Tell the encoder how long the sentence will be 
        x = Variable(torch.empty(batch_size, self.latent_size).normal_())
        x = torch.cat([len_one_hot, x], dim=1)
        
        x = self.linears(x).unsqueeze(1)
        h_n, c_n = self.recurrent(x)
        words = [h_n]

        for _ in range(sentence_len-1):
            h_n, c_n = self.recurrent(h_n, c_n) 
            words.append(h_n)

        return torch.cat(words, dim=1)


class Discriminator(nn.Module):
    def __init__(self, embed_size, hidden_size=16):
        super(Discriminator, self).__init__()

        self.embed_size = embed_size

        self.recurrent = nn.Sequential(
            nn.LSTM(embed_size, hidden_size, num_layers=3, batch_first=True), 
        )

        self.decider = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, (x, _) = self.recurrent(x)
        x = x[-1]

        return self.decider(x)
