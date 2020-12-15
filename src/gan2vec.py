import torch 

from torch import nn
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, latent_size, out_size, max_len=20, min_len=3, num_layers=1):
        super(Generator, self).__init__()

        self.out_size = out_size
        self.num_layers = num_layers
        self.MAX_LEN = max_len
        self.MIN_LEN = min_len 
        self.one_hot_size = max_len-min_len

        self.recurrent = nn.LSTM( 
            out_size+self.one_hot_size,
            out_size+self.one_hot_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.linear = nn.Linear(out_size+self.one_hot_size, out_size)

    '''
    Given batch of starter words, generate a sequence of outputs
    '''
    def forward(self, batch, sentence_len=5):
        h_n = Variable(
            torch.zeros(
                self.num_layers, 
                batch.size(0), 
                self.out_size+self.one_hot_size
            ).normal_()
        )

        c_n = Variable(
            torch.zeros(
                self.num_layers, 
                batch.size(0), 
                self.out_size+self.one_hot_size
            )#.normal_()
        )

        # Tell the encoder how long the sentence will be 
        one_hot = torch.zeros(batch.size(0), 1, self.one_hot_size)
        one_hot[:, :, self.MAX_LEN-sentence_len] = 1.0
        x_n = torch.cat([one_hot, batch], dim=-1)
        
        sentence = [batch]

        for _ in range(sentence_len):
            x_n, (h_n, c_n) = self.recurrent(x_n, (h_n, c_n))
            
            # Run output through one more linear layer w no activation
            x = x_n[:, 0, :]
            x = self.linear(x)
            sentence.append(x.unsqueeze(1))

        h_n = torch.cat(sentence, dim=1)
        return h_n

    def generate(self, batch, sentence_len=5):
        with torch.no_grad():
            return self.forward(batch, sentence_len=sentence_len)


class Discriminator(nn.Module):
    def __init__(self, embed_size, hidden_size=64):
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
        _, (_, x) = self.recurrent(x)
        x = x[-1]
        return self.decider(x)
