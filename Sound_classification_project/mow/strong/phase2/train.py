import os
import sys
import random
import numpy as np

from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint

from sklearn.utils import shuffle

base_path = '/tmp2/b03902110/newphase1'
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

def normalize(X_train, X_val, y_train):
    X = np.concatenate((X_train, X_val))

    filename = os.path.join(base_data_path, 'X_unverified.npy')
    X_un = np.load(filename)

    X = np.concatenate((X, X_un))

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X_verified = np.load('/tmp2/b03902110/X_train_ens_verified.npy')
    y_verified = np.load('/tmp2/b03902110/Y_train_ens_verified.npy')

    X_train = np.concatenate((X_train, X_verified))
    y_train = np.concatenate((y_train, y_verified))

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return X_train, X_val, y_train

if __name__ == '__main__':
    val_set_num = str(sys.argv[1])
    X, y = getTrainData()
    X_train, y_train, X_val, y_val = split_data(X, y, int(val_set_num))
    X_train, X_val, y_train = normalize(X_train, X_val, y_train)

    X_train, y_train = shuffle(X_train, y_train, random_state=0)

    model = load_model(sys.argv[2])
    model.summary()
    
    filename = os.path.join('/tmp2/b03902110/newphase2', 'full_10fold_model/model' + val_set_num)
    if not os.path.exists(filename):
        os.makedirs(filename)

    callback = ModelCheckpoint(filename + '/weights.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.h5', monitor='val_loss', save_best_only=False, period=1)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=128, epochs=50, callbacks=[callback])

    print('\n\n========== Done ==========\n\n')
    
    score = model.evaluate(X_val, y_val)
    print(score)

    # model.save(os.path.join(base_path, '10_fold_model/cnn_model' + val_set_num))
