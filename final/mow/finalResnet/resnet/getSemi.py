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
    # val_acc = [0.8032, 0.8059, 0.8005, 0.8194, 0.8167, 0.8302, 0.8248, 0.8221, 0.8086, 0.8059]
    # val_acc = [0.8275, 0.7978, 0.8194, 0.8544, 0.8356, 0.8275, 0.8598, 0.8518, 0.8437, 0.8437]
    # val_acc = [0.8598, 0.8518, 0.8491, 0.8787, 0.8787, 0.8410, 0.8518, 0.8625, 0.8518, 0.8787]
    val_acc = [0.8652, 0.8787, 0.8652, 0.8841, 0.8868, 0.8679, 0.8652, 0.8760, 0.8598, 0.8868]

    X_test, fname_test = getTestData()
    X_un, y_un, fname_un = getUnData()
    
    X_all = np.concatenate((X_un, X_test))
    fname_all = np.concatenate((fname_un, fname_test))

    npy_predict = os.path.join(base_path, 'npy_predict_self_cotrain_18')

    for i in range(10):
        filename = os.path.join(npy_predict, 'mow_mfcc4_resnet18_self_cotrain_unverified_{}.npy'.format(i+1))
        result = np.load(filename)

        if i == 0:
            score = result * val_acc[i]
        else:
            score += result * val_acc[i]

    val_acc_sum = sum(val_acc)
    score /= val_acc_sum

    print(np.sum(score, axis=1))

    topidx = []
    top = []

    for i in range(len(score)):
        tmp = np.argmax(score[i])
        topidx.append(tmp)
        top.append(score[i][tmp])

    topidx = np.array(topidx, dtype=int)
    top = np.array(top)

    mean = np.mean(top)
    std = np.std(top)
    threshold = mean + std
    threshold = 0.8

    count = 0

    f = open('./mow_mfcc4_resnet18_cotrain_Y3.csv', 'w')
    f.write('fname,label_verified\n')

    for i in range(len(score)):
        if top[i] >= threshold:
            count += 1
            f.write('{},{}\n'.format(fname_all[i], topidx[i]))

    f.close()
    print(topidx)
    print(top)
    print(mean, std, threshold, count)
