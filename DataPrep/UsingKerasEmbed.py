import pickle
from pathlib import Path

import keras.losses
from keras.models import Model
from keras.layers import Embedding, Dense, Input, Reshape, Flatten
from keras.optimizers import Adam


mypath = Path('/data/txtFiles/GutSubset')

EMBED_SIZE = 50
BATCH_SIZE = 128
EPOCHS = 40

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
    embed = Embedding(input_dim=7001,output_dim=EMBED_SIZE, name="embeding_layer")(input)
    # embed = Reshape((EMBED_SIZE,2))(embed)
    embed = Flatten()(embed)
    embed = Dense(128,activation='relu')(embed)
    embed = Dense(32, activation='relu')(embed)
    output = Dense(1,activation='sigmoid')(embed)

    model = Model(inputs= [input], outputs=output, name="TestModel")
    return model


model = model2_single_in()


opt = Adam(learning_rate=1e-2)
# loss = "binary_crossentropy"
# loss = keras.losses.CategoricalCrossentropy(from_logits=True)
loss = keras.losses.BinaryCrossentropy()
model.compile(loss=loss,optimizer=opt,metrics=['accuracy'],)
model.fit(x=X_train,y=y_train,epochs=EPOCHS,verbose=1,batch_size=BATCH_SIZE)
model.save(mypath/f"keras-embed-model-epoch-{EPOCHS}")
