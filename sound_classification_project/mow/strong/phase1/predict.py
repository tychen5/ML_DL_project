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

base_path = '/tmp2/b03902110/newphase1'
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

    fileX = os.path.join(base_data_path, 'X_unverified.npy')
    fileY = os.path.join(base_data_path, 'y_unverified.npy')
    filefname = os.path.join(base_data_path, 'fname_unverified.npy')

    X_un = np.load(fileX)
    y_un = np.load(fileY)
    fname_un = np.load(filefname)

    return X, y, fname, X_un, y_un, fname_un

def split_data(X, y, fname, X_un, y_un, fname_un, idx):
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

    if idx == -1:
        X_val = X_un
        y_val = y_un
        fname_val = fname_un

    return X_train, y_train, fname_train, X_val, y_val, fname_val

def normalize(X_train, X_val):
    X_all = np.concatenate((X_train, X_val))

    mean = np.mean(X_all, axis=0)
    std = np.std(X_all, axis=0)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return X_train, X_val

if __name__ == '__main__':
    fold_idx = int(sys.argv[1])
    X, y, fname, X_un, y_un, fname_un = getTrainData()
    X_train, y_train, fname_train, X_val, y_val, fname_val = split_data(X, y, fname, X_un, y_un, fname_un, fold_idx)
    X_train, X_val = normalize(X_train, X_val)

    base_model_path = os.path.join(base_path, '10_fold_model')
    model_name = sys.argv[2]
    filename = os.path.join(base_model_path, model_name)

    npy_predict = os.path.join(base_path, 'npy_predict_un')
    if not os.path.exists(npy_predict):
        os.makedirs(npy_predict)

    csv_predict = os.path.join(base_path, 'csv_predict_un')
    if not os.path.exists(csv_predict):
        os.makedirs(csv_predict)

    model = load_model(filename)

    print('Predicting X_val...')
    result = model.predict(X_val)
    np.save(os.path.join(npy_predict, 'mow_cnn2d_unverified_{}.npy'.format(sys.argv[3])), result)

    df = pd.DataFrame(result)
    df.insert(0, 'fname', fname_val)
    df.to_csv(os.path.join(csv_predict, 'mow_cnn2d_unverified_{}.csv'.format(sys.argv[3])), index=False, header=True)

    print('Evaluating X_val...')
    score = model.evaluate(X_val, y_val)
    print(score)

'''
print('Output Kaggle format...')
pred = np.argmax(result, axis=-1)
print(pred.shape)
label_pred = []
for i in pred:
    label_pred.append(map_reverse[i])
label_pred = np.array(label_pred)
print(label_pred.shape)
df = pd.DataFrame()
df.insert(0, 'fname', Y_fname_test)
df.insert(1, 'label', label_pred)
df.to_csv(os.path.join(kaggle_predict, 'mow_kaggle_{}.csv'.format(fold_num)), index=False, header=True)
'''
