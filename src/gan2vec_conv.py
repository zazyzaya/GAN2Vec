import torch 

from torch import nn
from torch.autograd import Variable

class ConvGenerator(nn.Module):
    def __init__(self, out_size, sentence_len=7, latent_size=8, hidden=64):
        super(ConvGenerator, self).__init__()

        self.out_size = out_size
        self.latent_size = latent_size
        self.hc = hidden
        self.sentence_len = sentence_len
        
        self.convs = nn.Sequential(
            nn.BatchNorm2d(self.hc),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.hc, self.hc, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.hc, self.hc//2, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.hc//2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.hc//2, sentence_len, 3, stride=1, padding=1)
        )

        self.lin = nn.Sequential(
           nn.Sigmoid(),
           nn.Linear((self.latent_size*4)**2, out_size) 
        )

    '''
    Generate output sentence (batch, sentence_len, dim)
    using input noise (batch, hidden, dim//4 ** (1/2), dim//4 ** (1/2))
    as if the noise were an image (I dunno, it works with images that way)
    '''
    def get_noise(self, batch_size):
        return Variable(
            torch.empty(
                batch_size, self.hc, self.latent_size, self.latent_size
            ).normal_()
        )
    
    def reshape_output(self, x):
        return x.view(
            x.size(0), self.sentence_len, (self.latent_size*4)**2
        )

    def lin3d(self, x):
        xn = []
        for i in range(x.size(1)):
            xn.append(self.lin(x[:, i, :]))

        return torch.stack(xn, dim=1)

    def forward(self, batch_size):
        z = self.get_noise(batch_size)
        x = self.convs(z)
        x = self.reshape_output(x)

        return self.lin3d(x)

    def generate(self, batch_size):
        with torch.no_grad():
            return self.forward(batch_size)


class ConvDiscriminator(nn.Module):
    def __init__(self, embed_size, hidden_size=64):
        super(ConvDiscriminator, self).__init__()
        raise NotImplementedError
