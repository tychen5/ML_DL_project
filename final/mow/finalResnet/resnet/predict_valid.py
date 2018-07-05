'''
###################################
Modified from Mike's predict_acc.py
###################################
'''

import os
import sys
import random
import pickle
import numpy as np
import pandas as pd

from keras.utils import to_categorical
from keras.models import load_model

base_path = '/home/tyt/how2ml/mfcc4'
base_data_path = os.path.join(base_path, 'data')
num_fold = 10

def getTrainData():
    X = []
    y = []
    fname = []

    for i in range(num_fold):
        fileX = os.path.join(base_data_path, 'X/X' + str(i+1) + '.npy')
        fileY = os.path.join(base_data_path, 'y/y' + str(i+1) + '.npy')
        filefname = os.path.join(base_data_path, 'fname/fname' + str(i+1) + '.npy')

        X.append(np.load(fileX))
        y.append(np.load(fileY))
        fname.append(np.load(filefname))

    X = np.array(X)
    y = np.array(y)
    fname = np.array(fname)

    return X, y, fname

def split_data(X, y, fname, idx):
    X_train = []
    y_train = []
    fname_train = []

    for i in range(num_fold):
        if i == idx:
            X_val = X[i]
            y_val = y[i]
            fname_val = fname[i]
            continue
        if X_train == []:
            X_train = X[i]
            y_train = y[i]
            fname_train = fname[i]
        else:
            X_train = np.concatenate((X_train, X[i]))
            y_train = np.concatenate((y_train, y[i]))
            fname_train = np.concatenate((fname_train, fname[i]))

    return X_train, y_train, fname_train, X_val, y_val, fname_val

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

if __name__ == '__main__':
    X_train_all, y_train_all, fname_train_all = getTrainData()
    X_test, fname_test = getTestData()
    
    result_all = []

    for i in range(10):
        _, _, _, X_all, _, fname_all = split_data(X_train_all, y_train_all, fname_train_all, i)

        base_model_path = os.path.join(base_path, 'cnn_model_152')
        model_name = 'model{}'.format(i)
        filename = os.path.join(base_model_path, model_name)

        npy_predict = os.path.join(base_path, 'final_npy_predict_phase3_val_152')
        if not os.path.exists(npy_predict):
            os.makedirs(npy_predict)

        csv_predict = os.path.join(base_path, 'final_csv_predict_phase3_val_152')
        if not os.path.exists(csv_predict):
            os.makedirs(csv_predict)

        model = load_model(filename)

        print('Predicting X_all...')
        result = model.predict(X_all)
        np.save(os.path.join(npy_predict, 'mow_mfcc4_resnet152_phase3_val_{}.npy'.format(i+1)), result)

        if result_all == []:
            result_all = result
        else:
            result_all = np.concatenate((result_all, result))


        df = pd.DataFrame(result)
        df.insert(0, 'fname', fname_all)
        df.to_csv(os.path.join(csv_predict, 'mow_mfcc4_resnet152_phase3_val_{}.csv'.format(i+1)), index=False, header=True)

    print(result_all.shape)
    print(fname_train_all.shape)

    fname_train_all = fname_train_all.reshape((-1, 1))

    df = pd.DataFrame(result_all)
    df.insert(0, 'fname', fname_train_all)
    df.to_csv(os.path.join(csv_predict, 'mow_mfcc4_resnet152_phase3_val_all.csv'), index=False, header=True)
