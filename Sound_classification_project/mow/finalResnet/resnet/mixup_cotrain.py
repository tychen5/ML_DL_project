import os
import sys
import pickle
import random
import numpy as np
import pandas as pd

from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

import resnet
from random_eraser import get_random_eraser
from mixup_generator import MixupGenerator

base_path = '/home/tyt/how2ml/mfcc4'
base_data_path = os.path.join(base_path, 'data')
num_fold = 10

def getTrainData():
    X = []
    y = []

    for i in range(num_fold):
        fileX = os.path.join(base_data_path, 'X/X' + str(i+1) + '.npy')
        fileY = os.path.join(base_data_path, 'y/y' + str(i+1) + '.npy')
        
        X.append(np.load(fileX))
        y.append(np.load(fileY))

    X = np.array(X)
    y = np.array(y)

    return X, y

def split_data(X, y, idx):
    X_train = []
    y_train = []
    
    for i in range(num_fold):
        if i == idx:
            X_val = X[i]
            y_val = y[i]
            continue
        if X_train == []:
            X_train = X[i]
            y_train = y[i]
        else:
            X_train = np.concatenate((X_train, X[i]))
            y_train = np.concatenate((y_train, y[i]))

    return X_train, y_train, X_val, y_val

def getUnData():
    fileX = os.path.join(base_data_path, 'X_unverified.npy')
    fileY = os.path.join(base_data_path, 'y_unverified.npy')
    filefname = os.path.join(base_data_path, 'fname_unverified.npy')

    X_un = np.load(fileX)
    y_un = np.load(fileY)
    fname_un = np.load(filefname)

    return X_un, y_un, fname_un

def getTestData():
    fileX = os.path.join(base_data_path, 'X_test.npy')
    filefname = os.path.join('./', 'fname_test.npy')

    X_test = np.load(fileX)
    fname_test = np.load(filefname)

    return X_test, fname_test

def normalize(X_train, X_val):
    filename = os.path.join(base_data_path, 'X_train.npy')
    X = np.load(filename)

    filename = os.path.join(base_data_path, 'X_test.npy')
    X_test = np.load(filename)

    X = np.concatenate((X, X_test))

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return X_train, X_val

def getSemiData():
    X_semi = []
    y_semi = []

    X_test, fname_test = getTestData()
    X_un, y_un, fname_un = getUnData()

    X_all = np.concatenate((X_un, X_test))
    fname_all = np.concatenate((fname_un, fname_test))

    semi_filename = os.path.join(base_data_path, 'Y_selftrain_ens_verified.csv')
    semi = pd.read_csv(semi_filename)

    with open('./map.pkl', 'rb') as f:
        map_dict = pickle.load(f)

    Y_dict = semi['label_verified'].map(map_dict)
    Y_dict = np.array(Y_dict)

    print(Y_dict)

    for i in range(len(semi)):
        tmpidx = np.argwhere(fname_all == semi['fname'][i])[0][0]
        X_semi.append(X_all[tmpidx])
        y_semi.append(to_categorical(Y_dict[i], num_classes=41))

    X_semi = np.array(X_semi)
    y_semi = np.array(y_semi)

    print(X_semi.shape)
    print(y_semi.shape)

    return X_semi, y_semi

if __name__ == '__main__':
    X_semi, y_semi = getSemiData()

    # fine tune
    for i in range(10):
        print('Fold {}'.format(i+1))
        val_set_num = str(i)
        X, y = getTrainData()
        X_train, y_train, X_val, y_val = split_data(X, y, int(val_set_num))
        # X_train, X_val = normalize(X_train, X_val)

        X_train = np.concatenate((X_train, X_semi))
        y_train = np.concatenate((y_train, y_semi))

        tmpidx = list(range(len(X_train)))
        random.shuffle(tmpidx)

        X_train = X_train[tmpidx]
        y_train = y_train[tmpidx]

        datagen = ImageDataGenerator(width_shift_range=0.05,
                                     height_shift_range=0.05,
                                     shear_range=0.084375,
                                     preprocessing_function=get_random_eraser(v_l=np.min(X_train),v_h=np.max(X_train)))
        
        training_generator = MixupGenerator(X_train, y_train, batch_size=32, alpha=0.5, datagen=datagen)()

        filename = os.path.join(base_path, 'phase3_mfcc4_resnet18_3/model' + val_set_num)
        if not os.path.exists(filename):
            os.makedirs(filename)

        callback = ModelCheckpoint(filename + '/weights.{epoch:04d}-{val_loss:.4f}-{val_acc:.4f}.h5', monitor='val_acc', save_best_only=True, verbose=1)
        early = EarlyStopping(monitor='val_acc', mode='max', patience=30, verbose=1)

        # model = resnet.ResnetBuilder.build_resnet_152((1, 20, 690), 41)
        modelfile = os.path.join(base_path, 'cnn_model_18/model' + val_set_num)
        model = load_model(modelfile)

        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.summary()

        model.fit_generator(generator=training_generator,
                            steps_per_epoch=(X_train.shape[0] // 32)*10,
                            validation_data=(X_val, y_val),
                            epochs=10000,
                            verbose=1,
                            callbacks=[callback, early])

        print('\n\n========== Done ==========\n\n')
    
