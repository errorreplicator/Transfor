import torch
from torchtext.legacy.data import Field, LabelField, TabularDataset, BucketIterator
from torchtext.legacy.datasets import IMDB
# from torchtext.datasets import IMDB
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# import spacy as sp
# spacy_eng = sp.load('en')

TEXT = Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
LABEL = LabelField(dtype=torch.float)


train, test = IMDB.splits(TEXT,LABEL)

# print(len(train),len(test))
# print(vars(train.examples[1])['text'])

# for x in range(5):
#     print(x,len(vars(train.examples[x])['text']),vars(train[x])['label'] ,vars(train.examples[x])['text'])


#  splits provide only training / test -> below creates validation dataset (and shuffle?)
train, validate = train.split(random_state = random.seed(1234), split_ratio = 0.8)


# for x in range(5):
#     print(x,len(vars(train.examples[x])['text']),vars(train[x])['label'] ,vars(train.examples[x])['text'])

# create vocabulary

TEXT.build_vocab(train)
LABEL.build_vocab(test)

print(TEXT.vocab)
# print(TEXT.vocab.freqs.most_common(20))
# print(TEXT.vocab.itos[:10])
print(TEXT.vocab.stoi['undirected'])
print(len(TEXT.vocab),len(LABEL.vocab))

BATCH_SIZE = 64


train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train, validate, test),
    batch_size = BATCH_SIZE,
    device = device)
