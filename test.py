import numpy as np


word_one_hot = np.zeros((1,10))
print(word_one_hot)
word_one_hot[0][3] = 1
print(word_one_hot)

def change_to_onehot(numbers_list,embed_size):
    word_one_hot = np.zeros((1,embed_size))
    print(word_one_hot.shape)
    context_one_hot = np.zeros((1, embed_size))
    word_one_hot[0][numbers_list[0]] = 1
    context_one_hot[0][numbers_list[1]] = 1
    return (word_one_hot,context_one_hot)


print(change_to_onehot([2,7],10))