import pickle
from pathlib import Path
from keras.models import load_model
from scipy.spatial import distance
import numpy as np

mypath = Path('/data/txtFiles/GutJustOne/')

with open(mypath/'tokenizer.pkltok','rb') as obj:
    tokenizer = pickle.load(obj)


model = load_model(mypath/'model_10.pkl')


# print(model.summary())

embed_vect = model.get_layer('word_lay_den_lin').get_weights()


embed_vect = model.get_layer('word_lay_den_lin').get_weights()
weights = embed_vect[0]


def topN_vect(word, tokenizer, topN, weights, topORbott='top'):
    my_words = []
    vectors = []
    my_word = word
    my_words.append(my_word)

    my_word_idx = tokenizer.word_index[my_word]

    my_vector = weights[my_word_idx]
    vectors.append(my_vector)

    distances = distance.cdist([my_vector], weights, "cosine")
    if topORbott == 'top':
        indexes = np.argpartition(distances[0], topN)[:topN]
        for idx in indexes:
            vectors.append(weights[idx])
            my_words.append(tokenizer.index_word[idx])

    else:
        indexes = np.argpartition(distances[0], topN)[topN:]
        for idx in indexes:
            vectors.append(weights[idx])
            my_words.append(tokenizer.index_word[idx])
    return (vectors, my_words)

word = 'king'
vectors, words = topN_vect(word,tokenizer,-4,weights,topORbott='bott')