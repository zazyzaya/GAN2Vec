import os 
import pandas as pd 

from gensim.models import Word2Vec

DATA_DIR = '~/Documents/Programming/Python/GAN2Vec/data'

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

    w2v = Word2Vec(sentences=haikus, size=embed_dim, sg=1, negative=20)
    return haikus, w2v

if __name__ == '__main__':
    h, w = load_haiku()

