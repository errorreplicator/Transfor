import pickle
from pathlib import Path
import numpy as np
from keras.models import Model
from keras.layers import Input,Dense, concatenate
from keras.utils import plot_model
from keras.optimizers import Adam
import tensorflow as tf
from sys import getsizeof
from sklearn.utils import shuffle
mypath = Path('/data/txtFiles/GutJustOne/')
# mypath = Path('/data/txtFiles/GutSubset')
# mypath = Path('/data/txtFiles/Gut1k')
# mypath = Path('/data/txtFiles/Gutenberg/txt')


with open(mypath/'X_Y_index_pairs.pklxy','rb') as file:
    (X_train,y_train) = pickle.load(file)

with open(mypath/'tokenizer.pkltok','rb') as file:
    tokenizer = pickle.load(file)

CORP_SIZE =  len(tokenizer.index_word)+1
EMBED_SIZE = 50
BATCH_SIZE = 512

# X_word = np.zeros((CORP_SIZE, EMBED_SIZE))
# print(round(getsizeof(X_word) / 1024 / 1024,2), " MBs")

def change_to_onehot(numbers_list, corp_size):
    word_one_hot = np.zeros((1, corp_size))
    context_one_hot = np.zeros((1, corp_size))
    if numbers_list[0] >= corp_size or numbers_list[1] >= corp_size:
        return  (word_one_hot,context_one_hot)
    word_one_hot[0][numbers_list[0]] = 1
    context_one_hot[0][numbers_list[1]] = 1
    return (word_one_hot,context_one_hot)


def data_generator (X_samples, y_samples, corp_size, batch_size=32, shuffle_data=True):
    num_samples = len(X_samples)
    while True:

        if shuffle_data:
            X_samples, y_samples = shuffle(X_samples, y_samples)



        for offset in range(0,num_samples,batch_size): # take 1st 32(batch size) pairs

            batch_X_samples = X_samples[offset:offset+batch_size]
            batch_y_samples = y_samples[offset:offset+batch_size]

            X_train_word = []
            X_train_context = []
            y_train = batch_y_samples

            for sample in batch_X_samples:
                X_word_vec, X_context_vec = change_to_onehot(sample, corp_size)

                X_train_word.append(X_word_vec)
                X_train_context.append(X_context_vec)

            X_train_word = np.array(X_train_word)
            X_train_context = np.array(X_train_context)
            y_train = np.array(y_train)

            yield [X_train_word,X_train_context],y_train

def data_generatorv2 (X_samples, y_samples, corp_size, batch_size=32, shuffle_data=True):
    num_samples = len(X_samples)
    one_hot_matrix = []
    for i in range(corp_size):
        one_hot_matrix.append([0 for x in range(corp_size)])

    for x in range(1, corp_size):
        one_hot_matrix[x][x] = 1

    # print(one_hot_matrix[:4])

    while True:

        if shuffle_data:
            X_samples, y_samples = shuffle(X_samples, y_samples)

        for offset in range(0,num_samples,batch_size): # take 1st 32(batch size) pairs

            batch_X_samples = X_samples[offset:offset+batch_size] # prepare batch for Word Context
            batch_y_samples = y_samples[offset:offset+batch_size] # batch for label

            X_train_word = []
            X_train_context = []
            y_train = batch_y_samples # yes I know it's not needed but for clarity ...

            for word_id_pair in batch_X_samples:

                X_train_word.append(one_hot_matrix[word_id_pair[0]]) # One Hot for Word
                X_train_context.append(one_hot_matrix[word_id_pair[1]]) # One Hot for Context

            X_train_word = np.array(X_train_word)
            X_train_context = np.array(X_train_context)
            y_train = np.array(y_train)

            yield [X_train_word,X_train_context],y_train


        # print(len(X_train))


def data_generatorv3 (X_samples, y_samples, corp_size, batch_size=32, shuffle_data=True):
    num_samples = len(X_samples)
    while True:

        if shuffle_data:
            X_samples, y_samples = shuffle(X_samples, y_samples)



        for offset in range(0,num_samples,batch_size): # take 1st 32(batch size) pairs

            batch_X_samples = X_samples[offset:offset+batch_size]
            batch_y_samples = y_samples[offset:offset+batch_size]

            X_train_word = []
            X_train_context = []
            y_train = batch_y_samples

            for sample in batch_X_samples:
                X_word_vec = tf.one_hot(indices=sample[0], depth=corp_size)
                X_context_vec = tf.one_hot(indices=sample[1],depth=corp_size)

                X_train_word.append(X_word_vec)
                X_train_context.append(X_context_vec)

            X_train_word = np.array(X_train_word)
            X_train_context = np.array(X_train_context)
            y_train = np.array(y_train)

            yield [X_train_word,X_train_context],y_train


train_generator = data_generatorv3(X_train,y_train,corp_size=CORP_SIZE,batch_size=BATCH_SIZE,shuffle_data=False)


input_word = Input(shape=(CORP_SIZE,),name="word_layer_input")
input_context = Input(shape=(CORP_SIZE,),name="context_layer_input")

word_layer = Dense(EMBED_SIZE, activation='linear', name="word_lay_den_lin")(input_word)
word_layer = Model(inputs = input_word, outputs=word_layer, name="word_layer_Model")

context_layer = Dense(EMBED_SIZE, activation='linear', name="context_lay_den_lin")(input_context)
context_layer = Model(inputs = input_context, outputs=context_layer,name='context_layer_Model')

combined = concatenate([word_layer.output, context_layer.output], name="concatenate")
output = Dense(1,activation='sigmoid',name="output_layer_1_sigmoid")(combined)

model = Model(inputs = [word_layer.input,context_layer.input],outputs = output,name="final_model")

# print(model.summary())
# plot_model(model, to_file=mypath/"model.png")

opt = Adam(learning_rate=1e-2)

model.compile(loss="binary_crossentropy",optimizer=opt,metrics=['accuracy'],)
epochs = 10
model.fit_generator(
   train_generator,
    epochs=epochs,
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    verbose=1
)

model.save(mypath/f"model_{epochs}.pkl")



#*****************************************
# print(10*" START ")
# X_word = tf.one_hot(X_train[:,[0]].reshape(len(X_train),),CORP_SIZE)
# X_context = tf.one_hot(X_train[:,[1]].reshape(len(X_train),),CORP_SIZE)
# print(10*" STOP ")


#*****************************************
# print(10*" START ")
#
# word_sample_onehot_list = X_train[:, [0]].reshape(len(X_train), )
# context_sample_onehot_list = X_train[:, [1]].reshape(len(X_train), )
# row_size = len(word_sample_onehot_list)
#
# X_word = np.zeros((row_size,CORP_SIZE))
# X_context = np.zeros((row_size,CORP_SIZE))
#
# for index, int_number in enumerate(word_sample_onehot_list):
#     X_word[index][int_number] = 1
#
# for index, int_number in enumerate(context_sample_onehot_list):
#     X_context[index][int_number] = 1
#
# print(10*" STOP ")
#*****************************************
# print(X_train[:10])
# print(type(X_train)) # save vocabulary and pass it here ???
# print(len(tokenizer.index_word))
# print(tokenizer.index_word[992])
# print(tokenizer.word_index['queen'])
