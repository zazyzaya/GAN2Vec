import os 
import pickle
import string 
import pandas as pd 

from gensim.models import Word2Vec

DATA_DIR = '../data'


def pipeline(s):
	s = ' '.join(s)
	s = s.replace('--', ' ')
	s = s.translate(str.maketrans('', '', string.punctuation))
	return s.split(' ')

def load_haiku(embed_dim=64):
	with open(os.path.join(DATA_DIR, 'clean_haiku.data'), 'rb') as f:
		haikus = pickle.load(f)

	for i in range(len(haikus)):
		haikus[i] = pipeline(haikus[i])
	
	with open(os.path.join('clean_haiku.data'), 'wb+') as f:
		pickle.dump(haikus, f, protocol=pickle.HIGHEST_PROTOCOL)

	w2v = Word2Vec(
		sentences=haikus, 
		size=embed_dim, 
		sg=1, 
		workers=32, 
		negative=20,
		min_count=1
	)

	w2v.save('w2v.model')
	return haikus, w2v

if __name__ == '__main__':
	h, w = load_haiku()

