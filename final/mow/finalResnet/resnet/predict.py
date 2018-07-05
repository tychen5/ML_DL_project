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
    X_test, fname_test = getTestData()
    # X_un, y_un, fname_un = getUnData()
    
    # X_all = np.concatenate((X_un, X_test))
    # fname_all = np.concatenate((fname_un, fname_test))

    X_all = X_test
    fname_all = fname_test

    for i in range(10):
        base_model_path = os.path.join(base_path, 'cnn_model_152')
        model_name = 'model{}'.format(i)
        filename = os.path.join(base_model_path, model_name)

        npy_predict = os.path.join(base_path, 'final_npy_predict_phase3_152')
        if not os.path.exists(npy_predict):
            os.makedirs(npy_predict)

        csv_predict = os.path.join(base_path, 'final_csv_predict_phase3_152')
        if not os.path.exists(csv_predict):
            os.makedirs(csv_predict)

        model = load_model(filename)

        print('Predicting X_all...')
        result = model.predict(X_all)
        np.save(os.path.join(npy_predict, 'mow_mfcc4_resnet152_phase3_test_{}.npy'.format(i+1)), result)

        df = pd.DataFrame(result)
        df.insert(0, 'fname', fname_all)
        df.to_csv(os.path.join(csv_predict, 'mow_mfcc4_resnet152_phase3_test_{}.csv'.format(i+1)), index=False, header=True)

