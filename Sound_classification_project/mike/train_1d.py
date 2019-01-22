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
                          GlobalMaxPool1D, Input, MaxPool1D, concatenate)
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
pd.options.mode.chained_assignment = None  # default='warn'

fold_path  = '/media/mike/87e285f0-29e5-4b1f-8037-d3715b9f90c5/ten_fold_data_raw'
if not os.path.exists(fold_path):
    os.mkdir(fold_path)


df = pd.read_csv('data/train_label.csv')

map_dict = pk.load(open('data/map.pkl' , 'rb'))
df_manu = df[df['manually_verified'] == 1]
df_manu['trans'] = df_manu['label'].map(map_dict)
df_manu['onehot'] = df_manu['trans'].apply(lambda x: to_categorical(x,num_classes=41))

np.load('X_train_verified_3710.npy')
fnames = df_manu['fname'].values

X =  np.load('X_train_verified_3710.npy')
Y =  df_manu['onehot'].tolist()
Y = np.array(Y)
Y = Y.reshape(-1 ,41)

X_train ,X_valid= X[:3339] , X[3339:] 
Y_train ,Y_valid= Y[:3339] , Y[3339:] 



def get_1d_conv_model(config):
    
    nclass = 41
    input_length = 352800
    
    inp = Input(shape=(input_length,1))
    x = Convolution1D(16, 9, activation=relu, padding="valid")(inp)
    x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
    x = MaxPool1D(16)(x)
    x = Dropout(rate=0.1)(x)
    
    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = MaxPool1D(4)(x)
    x = Dropout(rate=0.1)(x)
    
    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = MaxPool1D(4)(x)
    x = Dropout(rate=0.1)(x)
    
    x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
    x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(rate=0.2)(x)

    x = Dense(64, activation=relu)(x)
    x = Dense(1028, activation=relu)(x)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model


# kf = KFold(n_splits=10)
# k = 0

# # split manu data
# for train_index, test_index in kf.split(X):
#     k+=1
#     X_train , X_valid = X[train_index], X[test_index]
#     Y_train, Y_valid = Y[train_index], Y[test_index]
#     Y_valid_fname = fnames[test_index]
    
#     np.save( os.path.join(fold_path, 'X_train_{}_raw'.format(k)), X_train)
#     np.save( os.path.join(fold_path, 'Y_train_{}_raw'.format(k)), Y_train)
#     np.save( os.path.join(fold_path, 'X_valid_{}_raw'.format(k)), X_valid)
#     np.save( os.path.join(fold_path, 'Y_valid_{}_raw'.format(k)), Y_valid)
#     np.save( os.path.join(fold_path, 'valid_fname_{}_raw'.format(k)), Y_valid_fname)
#     print('{} fold split done'.format(k))

# print('verified raw data split done =====================')


