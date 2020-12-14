import torch 
import pickle 
import os 

from torch import nn 
from random import randint
from torch.optim import Adam 
from gan2vec import Discriminator, Generator
from torch.nn.utils.rnn import pack_padded_sequence
from gensim.models import Word2Vec

DATA_DIR = 'data'
IN_TEXT = 'clean_haiku.data'
IN_W2V  = 'w2v_haiku.model'

text = encoder = None 

def get_data():
    global text, encoder 
    if text:
        return 

    with open(os.path.join(DATA_DIR, IN_TEXT), 'rb') as f:
        text = pickle.load(f)
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
            try:
                sentence.append(torch.tensor(encoder.wv[w]))
            except:
                print(w)
                continue

        sentences.append(torch.stack(sentence).unsqueeze(0))

    print(len(sentences))

    # Pad input 
    d_size = sentences[0].size(2)
    for i in range(len(sentences)):
        sl = sentences[i].size(1)

        if sl < longest: 
            sentences[i] = torch.cat(
                [sentences[i], torch.zeros(1,longest-sl,d_size)], 
                dim=1
            )

    seq = torch.cat(sentences, dim=0)
    packer = pack_padded_sequence(
        seq, 
        seq_lens, 
        batch_first=True, 
        enforce_sorted=False
    )    

    return packer 

def train(epochs, batch_size=256, latent_size=64):
    get_data()
    num_samples = len(text)

    G = Generator(latent_size, 64)
    D = Discriminator(64)

    loss = nn.BCELoss()
    opt_g = Adam(G.parameters(), lr=0.001)
    opt_d = Adam(D.parameters(), lr=0.001)

    for e in range(epochs):
        start = randint(0, num_samples-batch_size-1)
        slen = randint(5,10)
        
        tl = torch.full((batch_size, 1), 1.0)
        fl = torch.zeros((batch_size, 1))

        # Train descriminator 
        opt_d.zero_grad() 
        real = get_lines(start, start+batch_size)
        fake = G(batch_size, sentence_len=slen)

        r_loss = loss(D(real), tl)
        f_loss = loss(D(fake), fl)

        r_loss.backward()
        f_loss.backward()
        d_loss = (r_loss.mean().item() + f_loss.mean().item()) / 2
        opt_d.step()

        # Train generator 
        opt_d.zero_grad() 
        fake = G(batch_size, sentence_len=slen)
        g_loss = loss(D(fake), tl)
        g_loss.backward() 
        opt_g.step() 

        g_loss = g_loss.item() 

        print('[%d] D Loss: %0.6f\tG Loss %0.6f' % (e, d_loss, g_loss))

        if e % 10 == 0:
            torch.save(G, 'generator.model')

if __name__ == '__main__':
    train(50)