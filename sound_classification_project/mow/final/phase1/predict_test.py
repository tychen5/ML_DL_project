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

base_data_path = os.path.join('/tmp2/b03902110/newphase1', 'data')

filename = os.path.join(base_data_path, 'X_train.npy')
X_train = np.load(filename)

filename = os.path.join(base_data_path, 'X_test.npy')
X_test = np.load(filename)

X_all = np.concatenate((X_train, X_test))

Y_test = pd.read_csv('./sample_submission.csv')
fname_test = Y_test['fname'].tolist()

def normalize(X_norm):
    mean = np.mean(X_all, axis=0)
    std = np.std(X_all, axis=0)

    X_norm = (X_norm - mean) / std

    return X_norm

if __name__ == '__main__':
    base_path = '/tmp2/b03902110/finalphase1'

    X_test = normalize(X_test)
    
    for i in range(10):
        base_model_path = os.path.join(base_path, 'cnn_model')
        model_name = 'model{}'.format(i)
        filename = os.path.join(base_model_path, model_name)

        npy_predict = os.path.join(base_path, 'npy_predict_test')
        if not os.path.exists(npy_predict):
            os.makedirs(npy_predict)

        csv_predict = os.path.join(base_path, 'csv_predict_test')
        if not os.path.exists(csv_predict):
            os.makedirs(csv_predict)

        model = load_model(filename)

        print('Predicting X_test...')
        result = model.predict(X_test)
        np.save(os.path.join(npy_predict, 'mow_cnn2d_test_{}.npy'.format(i+1)), result)

        df = pd.DataFrame(result)
        df.insert(0, 'fname', fname_test)
        df.to_csv(os.path.join(csv_predict, 'mow_cnn2d_test_{}.csv'.format(i+1)), index=False, header=True)

