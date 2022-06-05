from keras.preprocessing.text import Tokenizer, text_to_word_sequence, one_hot
import pandas as pd

test = ['to jest test','o co w tym wszystkim chodzi?']
mainTab = [
    "The future king is the prince Daughter is the princess",
    "Son is the prince",
    "Only a man can be a king",
    "Only a woman can be a queen",
    "The princess will be a queen",
    "Queen and king rule the realm",
    "The prince is a strong man",
    "The princess is a beautiful woman",
    "The royal family is the king and queen and their children",
    "Prince is only a boy now",
    "A boy will be a man"
]

columns = ['Message','Target']
ds = pd.read_csv('/data/imdb_reviews.csv',sep='\t',names=columns)

def get_tokens(listOfSentences):
    ret_table = []
    for sentence in listOfSentences:
        ret_table.append(text_to_word_sequence(sentence))
    return ret_table

tokenizer = Tokenizer()
tokenizer.fit_on_texts(mainTab)
word2index = tokenizer.word_index
index2word = tokenizer.index_word
docs_endoced = tokenizer.texts_to_sequences(mainTab)
max_len = max([len(sent) for sent in docs_endoced])
vocab_size = max_len


print(text_to_word_sequence(mainTab[0]))

#
# def getwordpairs(listOfSentences):
#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(listOfSentences)
#     word2index = tokenizer.word_index
#     index2word = tokenizer.index_word
#     docs_endoced = tokenizer.texts_to_sequences(listOfSentences)
#     max_len = max([len(sent) for sent in docs_endoced])
#     vocab_size = max_len



