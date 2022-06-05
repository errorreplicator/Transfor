import pickle
from pathlib import Path

from keras.models import Model
from keras.layers import Input,Dense, concatenate
from keras.utils import plot_model
from keras.optimizers import Adam
import tensorflow as tf

mypath = Path('/data/txtFiles/GutJustOne/')
# mypath = Path('/data/txtFiles/GutSubset')
# mypath = Path('/data/txtFiles/Gut1k')
# mypath = Path('/data/txtFiles/Gutenberg/txt')


with open(mypath/'X_Y_index_pairs.pklxy','rb') as file:
    (X_train,y_train) = pickle.load(file)

with open(mypath/'tokenizer.pkltok','rb') as file:
    tokenizer = pickle.load(file)

CORP_SIZE = len(tokenizer.index_word)
EMBED_SIZE = 100


input_word = Input(shape=(CORP_SIZE,),name="word_layer_input")
input_context = Input(shape=(CORP_SIZE,),name="context_layer_input")

word_layer = Dense(EMBED_SIZE, activation='linear', name="word_lay_den_lin")(input_word)
word_layer = Model(inputs = input_word, outputs=word_layer, name="word_layer_Model")

context_layer = Dense(EMBED_SIZE, activation='linear', name="context_lay_den_lin")(input_context)
context_layer = Model(inputs = input_context, outputs=context_layer,name='context_layer_Model')

combined = concatenate([word_layer.output, context_layer.output], name="concatenate")
output = Dense(1,activation='sigmoid',name="output_layer_1_sigmoid")(combined)

model = Model(inputs = [word_layer.input,context_layer.input],outputs = output,name="final_model")

opt = Adam(learning_rate=1e-3)

model.compile(loss="binary_crossentropy",optimizer=opt,metrics=['accuracy'])

model.fit(
    x=[tf.one_hot(X_train[0],CORP_SIZE),tf.one_hot(X_train[1],CORP_SIZE)],
    y=y_train,
    epochs=200,
    batch_size=128
)


# print(model.summary())
# plot_model(model, to_file=mypath/"model.png")

# print(X_train[:10])
# print(type(X_train)) # save vocabulary and pass it here ???
# print(len(tokenizer.index_word))
# print(tokenizer.index_word[992])
# print(tokenizer.word_index['queen'])
