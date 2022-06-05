import numpy as np
import tensorflow as tf
X_train = np.array([
                    [1,4],
                    [3,2],
                    [5,6],
                    [3,4],
                    [1,7],
                    [5,10]
                    ])

word = X_train[:,[0]].reshape(len(X_train),)
context = X_train[:,[1]].reshape(len(X_train),)

VOCAB_SIZE = 10
X_word = np.zeros((len(X_train),11))
for index, word_idx in enumerate(word):#X_train[:,[0]].reshape(6,)):
    print(word_idx)
    X_word[index][word_idx] = 1

print(X_word)



# X_word = []




# print(X_train[:,[0]].reshape(6,))
# print(tf.one_hot(X_train[:,[0]].reshape(6,),11))

