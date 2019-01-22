'''
reference :
https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data
'''
import os
import shutil

import numpy as np
import pickle as pk
import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



for i in range(1,11):
    X_train = np.load('data/ten_fold_data/X_train_{}.npy'.format(i)) 
    Y_train = np.load('data/ten_fold_data/Y_train_{}.npy'.format(i)) 
    X_test = np.load('data/ten_fold_data/X_valid_{}.npy'.format(i))
    Y_test = np.load('data/ten_fold_data/Y_valid_{}.npy'.format(i))

    X_train = X_train.reshape(3339, 13800)
    X_test = X_test.reshape(371 , 13800)

    reverse_dict = pk.load(open('data/map_reverse.pkl' , 'rb'))
    Y_train  = np.array([np.where(x >= 1.0) for x in Y_train]).reshape(3339,)
    Y_test = np.array([np.where(x >= 1.0) for x in Y_test]).reshape(371,)


    clf = SVC(probability = True)
    clf.fit(X_train, Y_train) 
    pred = clf.predict(X_test)

    acc_score = accuracy_score(Y_test, pred)
    print(acc_score)
    break

    