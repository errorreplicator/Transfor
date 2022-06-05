from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams
from pathlib import Path
import pickle
import numpy as np
# import time
import tensorflow as tf
import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

mypath = Path('/data/txtFiles/GutJustOne/')
# mypath = Path('/data/txtFiles/GutSubset')
# mypath = Path('/data/txtFiles/Gut1k')
# mypath = Path('/data/txtFiles/Gutenberg/txt')

with open(mypath/'1_master_sentence_ALL.pkl','rb') as file:
    (master_sentence,len_of_sent) = pickle.load(file)



SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
max_sent_len = 80
window_size = 2
max_vocab_size = 10000#12673


def get_sent_indexed(list_of_sentences):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list_of_sentences)
    sent_indexed = tokenizer.texts_to_sequences(list_of_sentences)
    return sent_indexed


def get_pos_samples(sentences_indexed):
    pos_samples = []
    for sentence in sentences_indexed:
        word_idx_pairs = skipgrams(
            sentence,
            vocabulary_size=max_vocab_size,
            window_size=window_size,
            negative_samples=0
        )
        pos_samples += word_idx_pairs[0]
    return pos_samples


def remove_duplicates(pos_sam_pairs):
    pos_samp_tup = set(tuple(i) for i in pos_sam_pairs)  # remove duplicates fast
    pos_samp_list = [list(i) for i in pos_samp_tup]  # pos samples back to list
    return (pos_samp_list)



def get_neg_samples(pos_sam_pairs,vocab_size,neg_sample_size):
    from random import randint
    i = 0 #neg_sample_count
    run_index = 0
    tmp = np.array(pos_sam_pairs)
    ds = pd.DataFrame(tmp,columns=['Word','Context']) # create dataframe for faster operations
    list_ofNegatives = []
    for index, row in ds.iterrows():
        run_index+=1
        word = row['Word'] # select centre word
        ds_sub = ds.loc[ds['Word'] ==word] # select all rows with centre word and it's context
        context_words_list = list(ds_sub['Context']) # type list of all context words
        while (i<neg_sample_size):
            neg_index = randint(1,vocab_size)
            if neg_index not in context_words_list:
                context_words_list.append(neg_index)
                list_ofNegatives.append([word,neg_index])
                i+=1
        if run_index % 10000 == 0:
            print(f"we are running row {run_index} / {ds.shape[0]}")

        # ds.drop(ds.loc[ds['Word']==word].index, inplace=True)
        i = 0

    return list_ofNegatives


sentence_indexed = get_sent_indexed(master_sentence)
pos_samples_pairs = get_pos_samples(sentence_indexed)
pos_samples_pairs = remove_duplicates(pos_samples_pairs)

pos_samples_pairs = pos_samples_pairs#[:10000] ##??????????????????????????????
print("starting neg sampling")
neg_samples_pairs = get_neg_samples(pos_samples_pairs,vocab_size=max_vocab_size,neg_sample_size=2)
print("just finished neg sampling")
len_pos = len(pos_samples_pairs)
len_neg = len(neg_samples_pairs)

pos_samples_pairs = np.array(pos_samples_pairs)
neg_samples_pairs = np.array(neg_samples_pairs)

pos_y_list = np.ones(shape=len_pos)
neg_y_list = np.zeros(shape=len_neg)

pos_neg_samp_pairs = np.concatenate((pos_samples_pairs, neg_samples_pairs),axis=0)
pos_neg_y = np.concatenate((pos_y_list, neg_y_list), axis=0)
print("beginning train test split")
# x_train, x_test, y_train, y_test = train_test_split(pos_neg_samp_pairs, pos_neg_y_pairs,test_size=0.1)
X_train, y_train = shuffle(pos_neg_samp_pairs,pos_neg_y)
print("beginning dump to pklxy")
with open(mypath / 'X_Y_index_pairs.pklxy', 'wb') as f:
    pickle.dump((X_train,y_train), f)
    print(f'just pickled X_Y_index_pairs.pklxy')
print(f"Train X size {len(X_train)} and Y {len(y_train)}")
# print(f"Test X size{len(x_test)} and Y {len(y_test)}")
print("ALL GOOD - goodby")









# print(pos_samples_pairs[-1],neg_samples_pairs[0],neg_samples_pairs[1])
# print(pos_neg_samp_pairs[9999],pos_neg_samp_pairs[10000])
# print(pos_neg_y_pairs[9999],pos_neg_y_pairs[10000])

# test_a = pos_neg_samp_pairs[9999]
# print(test_a)
# test_b = pos_neg_samp_pairs[10000]

# print(x_train)
#
# #
# result = np.where(x_train == test_a)
# print(result[0][0])
# # print(result)
# print(x_train[result[0][0]])
# print(y_train[result])

# print()
#
# result = np.where(x_train== test_b)
# print(x_train[result])
# print(y_train[result])


# print(100* "*")
# print(100* "*")
# print(pos_neg_samp_pairs.shape)
# print(pos_neg_y_pairs.shape)
# print(100* "*")

# np_array = np.array(pos_samples_pairs)
# ds = pd.DataFrame(pos_samples_pairs,columns=['Word','Context'])
# # print(ds.head(20))
# ds_sub = ds.loc[ds['Word'] == 2140]
# print(ds_sub)
# tmp_list = list(ds_sub['Context'])
# print(tmp_list)
#
# print(neg_samples_pairs[:30])
# print("len of pos samples: ",len(pos_samples_pairs))
# print("len of neg samples: ",len(neg_samples_pairs))


# t0 = time.time()
# a = 0
# for x in pos_samp_tup:
#     a = a + 1
# print(a)
# t1 = time.time()
#
# print("Loop time over SET: ",t1-t0)
#
# t0 = time.time()
# a = 0
# for x in pos_samp_list:
#     a = a + 1
# print(a)
# t1 = time.time()
#
# print("Loop time over LIST: ",t1-t0)

# All_pos_samples = []
# for pair in pos_samples:
#     if pair[0] != pair[1] and (pair not in All_pos_samples):
#         All_pos_samples.append(pair)
#
# print(len(All_pos_samples))

# owerall pairs = 1403658; duplicates = 1403658


# counter = 0
# for pair in ALL_pos_samples:
#     # print(pair, pair[0],pair[1])
#     if pair[0] == pair[1]:
#         counter+=1
#         print(pair," ",tokenizer.index_word[pair[0]]," ",tokenizer.index_word[pair[1]]," ",counter)
# 7920

# print(master_pos_samp[2807300:2807314])

# for idx, element in enumerate(master_pos_samp):
#     if isinstance(element,int):
#         print(element,idx)



# print(len(positive_samples[0][:]))
# sent_encod = text_to_word_sequence(master_sentence)
# # for sentense in master_sentence:
# #     sent_encod = tokenizer.tex(master_sentence)
#
#
# print(sent_encod[:10])


# for sent in master_sentence[22:33]:
#     print(sent)
#
# for no in len_of_sent[22:33]:
#     print(no)

# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(master_sentence)
# word2index = tokenizer.word_index
# index2word = tokenizer.index_word
#
# sent_encoded = tokenizer.texts_to_sequences(master_sentence)
# sentences_len = [len(sent) for sent in ma]

