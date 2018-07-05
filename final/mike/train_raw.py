import os
import shutil
import numpy as np
import pickle as pk
import pandas as pd
from keras.utils import to_categorical ,Sequence
from keras import losses, models, optimizers
from keras.models import Sequential
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.layers import (Convolution1D, Dense, Dropout, GlobalAveragePooling1D,
                          GlobalMaxPool1D , MaxPooling1D, Input, MaxPool1D, concatenate)
from keras.layers import Conv1D, Conv2D
from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten,
                          GlobalMaxPool2D, MaxPool2D, concatenate, Activation , MaxPooling2D)
from keras.layers import Activation, LeakyReLU
from keras import backend as K

import pandas as pd 
import numpy as np 
from sklearn.model_selection import KFold
import pickle as pk 
import os

from keras.utils import to_categorical ,Sequence
import resnet

pd.options.mode.chained_assignment = None  # default='warn'

# gpu usage limit ==============================================================

# import tensorflow as tf
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# tf.keras.backend.set_session(sess)



audio_martix = np.load('data/raw/X_train.npy')
df = pd.read_csv('data/train_label.csv')

map_dict = pk.load(open('data/map.pkl' , 'rb'))

df_manu = df[df['manually_verified'] == 1]
df_manu['trans'] = df_manu['label'].map(map_dict)
df_manu['onehot'] = df_manu['trans'].apply(lambda x: to_categorical(x,num_classes=41))

manu_veri_idx = df_manu.index.values
fnames = df_manu['fname'].values

X = audio_martix[manu_veri_idx]
Y = df_manu['onehot'].tolist()
Y = np.array(Y)
Y = Y.reshape(-1 ,41)

X = X.reshape(3710,88200,1)

print(X.shape)
print(Y.shape)


X_train ,X_valid= X[:3339] , X[3339:] 
Y_train ,Y_valid= Y[:3339] , Y[3339:] 

MODEL_FOLDER = 'model_full'

if not os.path.exists(MODEL_FOLDER):
    os.mkdir(MODEL_FOLDER)

# 0.66038
def get_1d_conv_model():
    
    nclass = 41
    input_length = 88200
    
    inp = Input(shape=(input_length,1))
    x = Convolution1D(16, 9, activation=relu, padding="valid")(inp)
    #x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
    x = MaxPool1D(16)(x)
    x = Dropout(rate=0.3)(x)
    
    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    #x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = MaxPool1D(4)(x)
    x = Dropout(rate=0.3)(x)
    
    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    #x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = MaxPool1D(4)(x)
    x = Dropout(rate=0.3)(x)
    
    x = Convolution1D(64, 3, activation=relu, padding="valid")(x)
    #x = Convolution1D(64, 3, activation=relu, padding="valid")(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(rate=0.3)(x)

    x = Dense(64, activation=relu)(x)
    x = Dense(1028, activation=relu)(x)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model
    '''
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(88200, 1)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))

    model.add(Dense(256,activation=relu))
    model.add(Dense(512,activation=relu))

    model.add(Dropout(0.3))

    model.add(Dense(41, activation='softmax'))

    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model
    '''





checkpoint = ModelCheckpoint('model_full/best_{val_acc:.5f}.h5', monitor='val_acc', verbose=1, save_best_only=True)
early = EarlyStopping(monitor="val_acc", mode="max", patience=50)
callbacks_list = [checkpoint, early]

model = get_1d_conv_model()


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()


history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), callbacks=callbacks_list,
                        batch_size=20, epochs=10000)
