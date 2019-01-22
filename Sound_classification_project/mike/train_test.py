'''
reference :
https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data
'''
import os
from os import listdir
from os.path import isfile, join
import shutil

import numpy as np
import pickle as pk
import pandas as pd


from keras.utils import to_categorical ,Sequence
from keras import losses, models, optimizers
from keras.models import Sequential
from keras.activations import relu, softmax
from keras.models import load_model
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.layers import (Convolution1D, Dense, Dropout, GlobalAveragePooling1D,
                          GlobalMaxPool1D, Input, MaxPool1D, concatenate)

from keras.layers import Conv1D, Conv2D
from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten,
                          GlobalMaxPool2D, MaxPool2D, concatenate, Activation , MaxPooling2D)
from keras.layers import Activation, LeakyReLU
from keras import backend as K

from sklearn.model_selection import KFold
from sklearn.utils import shuffle

import resnet


map_dict = pk.load(open('data/map.pkl' , 'rb'))


model_path = 'train_test_verified'

if not os.path.exists(model_path):
    os.mkdir(model_path)


# refine_path = 'cnn2d_verified_refine'

# models = [join(model_path, f) for f in listdir(model_path) if isfile(join(model_path, f))]

# if not os.path.exists(refine_path):
#     os.mkdir(refine_path)

df = pd.read_csv('data/train_label.csv') 
df['trans'] = df['label'].map(map_dict)
df['onehot'] = df['trans'].apply(lambda x: to_categorical(x,num_classes=41))
Y =  df_manu['onehot'].tolist()
Y = np.array(Y)
Y = Y.reshape(-1 ,41)

X = np.load('data/mfcc/X_train.npy') 

X , Y = shuffle(X, Y, random_state=5)

split = X.shape[0] *(9/10)
X_train , X_test = X[:split] , X[split:]
Y_train , Y_test = Y[:split] , Y[split:]

print( X_train.shape)
    


checkpoint = ModelCheckpoint(os.path.join(model_path,'best_%d_{val_acc:.5f}.h5'%i), monitor='val_acc', verbose=1, save_best_only=True)
early = EarlyStopping(monitor="val_acc", mode="max", patience=40)
callbacks_list = [checkpoint, early]