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

from sklearn.metrics import accuracy_score

with open('map.pkl', 'rb') as f:
    map_dict = pickle.load(f)

Y_test = pd.read_csv('/tmp2/b03902110/data/test/test_Y.csv')
Y_dict = Y_test['label'].map(map_dict)
Y_dict = np.array(Y_dict)
print(Y_dict.shape)
print(Y_dict)

Y_fname = Y_test['fname'].tolist()

Y_all = []
for i in Y_dict:
    Y_all.append(to_categorical(i, num_classes=41))
Y_all = np.array(Y_all)
print(Y_all)
print(Y_all.shape)

X_train = np.load('/tmp2/b03902110/data/train/train3_X.npy')
X_test = np.load('/tmp2/b03902110/data/test/test_X.npy')

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_test = (X_test - mean) / std

acc = {}

predict_npy_folder = './npy_predict'
predict_csv_folder = './csv_predict'

score = 0.0

for i in range(1, 7):
    print('round: {}'.format(i))

    model = load_model('/tmp2/b03902110/cnn_model{}'.format(i))
    result = model.predict(X_test)
    np.save(os.path.join(predict_npy_folder, 'mow_predict_{}.npy'.format(i)), result)

    df = pd.DataFrame(result)
    df.insert(0, 'fname', Y_fname)
    df.to_csv(os.path.join(predict_csv_folder, 'mow_predict_{}.csv'.format(i)), index=False, header=True)

    pred = np.argmax(result, axis=-1)
    acc = accuracy_score(Y_dict, pred)
    print('fold{} accuracy: {}'.format(i, acc))

    score += result

pred = np.argmax(score, axis=-1)
acc = accuracy_score(Y_dict, pred)
print('final accuracy: {}'.format(acc))
