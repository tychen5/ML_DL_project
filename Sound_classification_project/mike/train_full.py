'''
reference :
https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data
'''
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
from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K

from sklearn.model_selection import KFold

from random_eraser import get_random_eraser

from keras.optimizers import Adam
import resnet
from model import model_mlt_cnn_alexnet
# gpu usage limit ==============================================================
'''
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(sess)
'''
# category map dict =====================================================
map_dict = pk.load(open('data/map.pkl' , 'rb'))


MODEL_FOLDER = 'model_full_resnet2'

if not os.path.exists(MODEL_FOLDER):
    os.mkdir(MODEL_FOLDER)

def get_2d_conv_model():

    model = Sequential()

    # first layer need input shape
    # random 64 feature detector
    model.add(Conv2D(64, kernel_size=(5, 5), input_shape=(40, 345,1), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=0.03))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=0.03))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=0.03))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.35))

    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=0.03))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.4))

    model.add(Flatten())

    # fully connected layer 
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(41, activation='softmax', kernel_initializer='glorot_normal'))
    return model


kf = KFold(n_splits=10)


all_x = np.concatenate( (np.load('data/mfcc/X_train.npy') , np.load('data/X_test.npy')))

for i in range(8,11):

    X_train = np.load('data/ten_fold_data/X_train_{}.npy'.format(i)) 
    Y_train = np.load('data/ten_fold_data/Y_train_{}.npy'.format(i)) 
    X_test = np.load('data/ten_fold_data/X_valid_{}.npy'.format(i))
    Y_test = np.load('data/ten_fold_data/Y_valid_{}.npy'.format(i))

    # normalize
    # X_train = (X_train - mean)/std
    # X_test = (X_test - mean)/std
    
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    
    checkpoint = ModelCheckpoint(os.path.join(MODEL_FOLDER,'best_%d_{val_acc:.5f}.h5'%i), monitor='val_acc', verbose=1, save_best_only=True)
    early = EarlyStopping(monitor="val_acc", mode="max", patience=80)
    callbacks_list = [checkpoint, early]

    print("#"*50)
    print("Fold: ", i)

    model = resnet.ResnetBuilder.build_resnet_18((1, 40, 345), 41)


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


    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        preprocessing_function=get_random_eraser(v_l=np.min(all_x), v_h=np.max(all_x)) # Trainset's boundaries.
    )


    mygenerator = MixupGenerator(X_train, Y_train, alpha=1.0, batch_size=128, datagen=datagen)


    # ======================================================================================================
    
    model.compile(loss='categorical_crossentropy',
             optimizer=Adam(lr=0.0001),
             metrics=['accuracy'])

    # history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),  callbacks=callbacks_list, batch_size = 64, epochs=10000)
    
    history = model.fit_generator(mygenerator(),
                    steps_per_epoch= 10*X_train.shape[0] // 128,
                    epochs=10000,
                    validation_data=(X_test,Y_test), callbacks=callbacks_list)


    

    

