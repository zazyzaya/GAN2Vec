import torch
import os 

from random import randint
from gensim.models import Word2Vec
from train import DATA_DIR, IN_W2V, get_lines
from gan2vec import Generator

encoder = Word2Vec.load(os.path.join(DATA_DIR, IN_W2V)).wv
G = torch.load('generator.model')

ipt = ''
while('q' not in ipt):
    rnd = randint(0, 256)
    _, sw = get_lines(rnd, rnd+2)

    s = G.generate(sw)
    s = s[0].detach().numpy()
    print(s.shape)

    st = [
        encoder.most_similar([s[i]], topn=1)[0]
        for i in range(s.shape[0])
    ]

    st, sim = list(zip(*st))

    print(' '.join(st))
    print('\t'.join(['%0.4f' % i for i in sim]))
    #print(s)
    ipt = input()