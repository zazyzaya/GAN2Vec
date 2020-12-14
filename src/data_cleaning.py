import os 
import pickle
import pandas as pd 

from gensim.models import Word2Vec

DATA_DIR = '../data'

def load_haiku(embed_dim=64):
	df = pd.read_csv(os.path.join(DATA_DIR, 'all_haiku.csv'))
	df = df['0'] + ' ' + df['1'] + ' ' + df['2']

	hdirty = list(df)
	haikus = []
	for h in hdirty:
		try:
			haikus.append(
				h.lower().replace('  ', ' ').split(' ')
			)
		except: # Quick n dirty way to fix nan errors 
			pass	

	del hdirty 
	del df 

	with open('clean_haiku.data', 'wb+') as outf:
		pickle.dump(haikus, outf, pickle.HIGHEST_PROTOCOL)

	w2v = Word2Vec(
		sentences=haikus, 
		size=embed_dim, 
		sg=1, 
		workers=32, 
		negative=20
	)

	w2v.save('w2v.model')
	return haikus, w2v

if __name__ == '__main__':
	h, w = load_haiku()

