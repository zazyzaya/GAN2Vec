import torch
import os 

from random import randint
from gensim.models import Word2Vec
from train import DATA_DIR, IN_W2V
from gan2vec import Generator

encoder = Word2Vec.load(os.path.join(DATA_DIR, IN_W2V)).wv
G = torch.load('generator.model')

ipt = ''
while('q' not in ipt):
    s = G(2, sentence_len=randint(5,10))
    s = s[0].detach().numpy()
    print(s.shape)

    s = [
        encoder.most_similar([s[i]], topn=1)[0][0]
        for i in range(s.shape[0])
    ]

    print(' '.join(s))
    ipt = input()