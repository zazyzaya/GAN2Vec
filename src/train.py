import torch 
import pickle 
import os 
import time 

from torch import nn 
from torch.autograd import Variable 
from random import randint
from torch.optim import Adam 
from gan2vec import Discriminator, Generator
#from gan2vec_conv import ConvGenerator
from torch.nn.utils.rnn import pack_padded_sequence
from gensim.models import Word2Vec

DATA_DIR = 'data'
#DATA_DIR = 'code/GAN2Vec/data' # For debugger
IN_TEXT = 'cleaned_haiku.data'
IN_W2V  = 'w2v_haiku.model'

text = encoder = None 

def get_data():
    global text, encoder 
    if text:
        return 

    with open(os.path.join(DATA_DIR, IN_TEXT), 'rb') as f:
        text = pickle.load(f)[:256]
    encoder = Word2Vec.load(os.path.join(DATA_DIR, IN_W2V))


def get_lines(start,end):
    get_data() 

    seq_lens = []
    sentences = []
    longest = 0
    for l in text[start:end]:
        seq_lens.append(len(l))
        longest = len(l) if len(l) > longest else longest 

        sentence = []        
        for w in l:
            sentence.append(torch.tensor(encoder.wv[w]))
            
        sentences.append(torch.stack(sentence).unsqueeze(0))

    # Pad input 
    d_size = sentences[0].size(2)
    for i in range(len(sentences)):
        sl = sentences[i].size(1)

        if sl < longest: 
            sentences[i] = torch.cat(
                [sentences[i], torch.zeros(1,longest-sl,d_size)], 
                dim=1
            )

    # Need to squish sentences into [0,1] domain
    seq = torch.cat(sentences, dim=0)
    #seq = torch.sigmoid(seq)
    start_words = seq[:, 0:1, :]
    packer = pack_padded_sequence(
        seq, 
        seq_lens, 
        batch_first=True, 
        enforce_sorted=False
    )    

    return packer , start_words

def get_closest(sentences):
    scores = []
    wv = encoder.wv
    for s in sentences.detach().numpy():
        st = [
            wv[wv.most_similar([s[i]], topn=1)[0][0]]
            for i in range(s.shape[0])
        ]
        scores.append(torch.tensor(st))

    return torch.stack(scores, dim=0)

def train(epochs, batch_size=256, latent_size=256, K=1):
    get_data()
    num_samples = len(text)

    G = Generator(64, 64)
    D = Discriminator(64)

    l2 = nn.MSELoss()
    loss = nn.BCELoss()
    opt_d = Adam(D.parameters(), lr=0.002, betas=(0.5, 0.999))
    opt_g = Adam(G.parameters(), lr=0.002, betas=(0.5, 0.999))

    for e in range(epochs):
        i = 0
        while batch_size*i < num_samples:
            stime = time.time()
            
            start = batch_size*i
            end = min(batch_size*(i+1), num_samples)
            bs = end-start

            # Use lable smoothing
            tl = torch.full((bs, 1), 0.9)
            fl = torch.full((bs, 1), 0.1)

            # Train descriminator 
            opt_d.zero_grad() 
            real, greal = get_lines(start, end)
            fake = G(greal)

            r_loss = loss(D(real), tl)
            f_loss = loss(D(fake), fl)

            r_loss.backward()
            f_loss.backward()
            d_loss = (r_loss.mean().item() + f_loss.mean().item()) / 2
            opt_d.step()

            # Train generator 
            for _ in range(K):
                opt_g.zero_grad() 

                # GAN fooling ability
                fake = G(greal) 
                g_loss = loss(D(fake), tl)
                g_loss.backward()
                opt_g.step() 
                
            g_loss = g_loss.item() 

            print(
                '[%d] D Loss: %0.3f  G Loss %0.3f  (%0.1fs)' % 
                (e, d_loss, g_loss, time.time()-stime)
            )

            i += 1

        if e % 10 == 0:
            torch.save(G, 'generator.model')

    torch.save(G, 'generator.model')

torch.set_num_threads(16)
if __name__ == '__main__':
    train(1000, batch_size=256)
