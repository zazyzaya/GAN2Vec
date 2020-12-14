import torch
import os 

from random import randint
from gensim.models import Word2Vec
from train import DATA_DIR, IN_W2V
from gan2vec import Generator

encoder = Word2Vec.load(os.path.join(DATA_DIR, IN_W2V))
G = torch.load('generator.model')

while(True):
    s = G(2, sentence_len=randint(5,10))
    s = s[0]