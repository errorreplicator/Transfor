import pickle
from pathlib import Path

import keras.losses
from keras.models import Model
from keras.layers import Embedding, Dense, Input, Reshape, Flatten, concatenate
from keras.optimizers import Adam


mypath = Path('/data/txtFiles/GutSubset')

EMBED_SIZE = 50
BATCH_SIZE = 128
EPOCHS = 20
CORP_SIZE = 7000+1

with open(mypath/'X_Y_index_pairs.pklxy','rb') as file:
    (X_train,y_train) = pickle.load(file)

with open(mypath/'tokenizer.pkltok','rb') as file:
    tokenizer = pickle.load(file)

def model1_single_in():

    input = Input(shape=(2,))
    embed = Embedding(input_dim=7001,output_dim=EMBED_SIZE, name="embeding_layer")(input)
    # embed = Reshape((EMBED_SIZE,2))(embed)
    embed = Flatten()(embed)
    output = Dense(1,activation='sigmoid')(embed)

    model = Model(inputs= [input], outputs=output, name="TestModel")
    return model

def model2_single_in():

    input = Input(shape=(2,))
    embed = Embedding(input_dim=CORP_SIZE,output_dim=EMBED_SIZE, name="embeding_layer")(input)
    # embed = Reshape((EMBED_SIZE,2))(embed)
    embed = Flatten()(embed)
    embed = Dense(128,activation='relu')(embed)
    embed = Dense(32, activation='relu')(embed)
    output = Dense(1,activation='sigmoid')(embed)

    model = Model(inputs= [input], outputs=output, name="TestModel")
    return model


def model3_single_in():
    input_word = Input(shape=(1,), name="word_layer_input")
    input_context = Input(shape=(1,), name="context_layer_input")

    word_layer = Embedding(input_dim=CORP_SIZE, output_dim=EMBED_SIZE, name="embeding_layer_target")(input_word)
    # word_layer = Dense(EMBED_SIZE, activation='linear', name="word_lay_den_lin")(input_word)
    word_layer = Flatten()(word_layer)
    word_layer = Model(inputs=input_word, outputs=word_layer, name="word_layer_Model")

    context_layer = Embedding(input_dim=CORP_SIZE, output_dim=EMBED_SIZE, name="embeding_layer_context")(input_context)
    # context_layer = Dense(EMBED_SIZE, activation='linear', name="context_lay_den_lin")(input_context)
    context_layer = Flatten()(context_layer)
    context_layer = Model(inputs=input_context, outputs=context_layer, name='context_layer_Model')

    combined = concatenate([word_layer.output, context_layer.output], name="concatenate")
    output = Dense(1, activation='sigmoid', name="output_layer_1_sigmoid")(combined)

    model = Model(inputs=[word_layer.input, context_layer.input], outputs=output, name="final_model")

    return model

model = model3_single_in()


opt = Adam(learning_rate=1e-2)
# loss = "binary_crossentropy"
# loss = keras.losses.CategoricalCrossentropy(from_logits=True)
loss = keras.losses.BinaryCrossentropy()
model.compile(loss=loss,optimizer=opt,metrics=['accuracy'],)
model.fit(x=[X_train[0],X_train[1]],y=y_train,epochs=EPOCHS,verbose=1,batch_size=BATCH_SIZE)
model.save(mypath/f"keras-embed-model-epoch-{EPOCHS}")
