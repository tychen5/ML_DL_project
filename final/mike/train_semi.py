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
from random_eraser import get_random_eraser
from keras.preprocessing.image import ImageDataGenerator
import resnet

map_dict = pk.load(open('data/map.pkl' , 'rb'))

model_path = 'model_full_resnet18_gen'
refine_path = 'model_full_resnet18_gen_refine'


if not os.path.exists(refine_path):
    os.mkdir(refine_path)


X_train_semi = np.load('data/mfcc/X_train_ens_verified.npy')
df = pd.read_csv('data/mfcc/Y_train_ens_verified.csv')
df['trans'] = df['label_verified'].map(map_dict)
df['onehot'] = df['trans'].apply(lambda x: to_categorical(x,num_classes=41))

Y_train_semi =  df['onehot'].tolist()
Y_train_semi = np.array(Y_train_semi)
Y_train_semi = Y_train_semi.reshape(-1 ,41)


 # data generator ====================================================================================
class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y

# mean , std = np.load('data/mean_std.npy')

# for i , m  in enumerate(models):
for i in range(1,11):
    
    X_train = np.load('data/ten_fold_data/X_train_{}.npy'.format(i)) 
    Y_train = np.load('data/ten_fold_data/Y_train_{}.npy'.format(i)) 
    X_test = np.load('data/ten_fold_data/X_valid_{}.npy'.format(i))
    Y_test = np.load('data/ten_fold_data/Y_valid_{}.npy'.format(i))

    print('verified data:')
    print(X_train.shape)
    print(Y_train.shape)
    print('semi data')
    print(X_train_semi.shape)
    print(Y_train_semi.shape)
    
    
    #append semi data 
    X_train = np.append(X_train,X_train_semi , axis=0)
    Y_train = np.append(Y_train,Y_train_semi , axis=0)
    
    # shuffle new data
    X_train , Y_train = shuffle(X_train, Y_train, random_state=0) 


    print('after append semi data:')
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    model = load_model(join(model_path,'best_{}.h5'.format(i)))

    checkpoint = ModelCheckpoint(join(refine_path , 'best_semi_%d_{val_acc:.5f}.h5'%i), monitor='val_acc', verbose=1, save_best_only=True)
    early = EarlyStopping(monitor="val_acc", mode="max", patience=60)

    print("#"*50)
    print("Fold: ", i)

   
    
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        preprocessing_function=get_random_eraser(v_l=np.min(X_train), v_h=np.max(X_train)) # Trainset's boundaries.
    )

    mygenerator = MixupGenerator(X_train, Y_train, alpha=1.0, batch_size=128, datagen=datagen)

    history = model.fit_generator(mygenerator(),
                    steps_per_epoch= X_train.shape[0] // 128,
                    epochs=10000,
                    validation_data=(X_test,Y_test), callbacks=[checkpoint, early])

    

