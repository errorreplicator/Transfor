import torch
from torchtext.legacy.data import Field, LabelField, BucketIterator
from torchtext.legacy.datasets import IMDB
import random

TEXT = Field(tokenize='spacy',tokenizer_language='en_core_web_sm')
LABEL = LabelField(dtype=torch.float)

train, test = IMDB.splits(TEXT,LABEL)


# for x in range(5):
#     print(x,len(vars(train.examples[x])['text']),vars(train[x])['label'] ,vars(train.examples[x])['text'])

train, validate = train.split(random_state = random.seed(1234), split_ratio = 0.8)

TEXT.build_vocab(train,max_size=10000)
LABEL.build_vocab(test)

# print(TEXT.vocab)
# print(TEXT.vocab.freqs.most_common(20))
print(TEXT.vocab.itos[:10])
print(TEXT.vocab.itos[22])
print(TEXT.vocab.stoi['movie'])
# print(TEXT.vocab.stoi['undirected'])
# print(len(TEXT.vocab),len(LABEL.vocab))
