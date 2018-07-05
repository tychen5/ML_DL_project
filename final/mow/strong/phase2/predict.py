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

with open('map_reverse.pkl', 'rb') as f:
    map_reverse = pickle.load(f)

Y_train = pd.read_csv('/tmp2/b03902110/phase2/data/train_label.csv')
Y_dict = Y_train['label'].map(map_dict)
Y_dict = np.array(Y_dict)
print(Y_dict.shape)
print(Y_dict)

Y_fname_train = Y_train['fname'].tolist()

Y_test = pd.read_csv('./sample_submission.csv')
Y_fname_test = Y_test['fname'].tolist()

Y_all = []
for i in Y_dict:
    Y_all.append(to_categorical(i, num_classes=41))
Y_all = np.array(Y_all)
print(Y_all)
print(Y_all.shape)

X_train = np.load('/tmp2/b03902110/phase2/data/X_train.npy')
X_test = np.load('/tmp2/b03902110/phase2/data/X_test.npy')

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

base = '/tmp2/b03902110/newphase2'
modelbase = os.path.join(base, '10_fold_model')
name = sys.argv[1]
fold_num = int(sys.argv[2])
filename = os.path.join(modelbase, name)

X_val = np.load('/tmp2/b03902110/newphase1/data/X/X{}.npy'.format(fold_num+1))
X_val = (X_val - mean) / std

Y_val = np.load('/tmp2/b03902110/newphase1/data/y/y{}.npy'.format(fold_num+1))

npy_predict = os.path.join(base, 'npy_predict')
if not os.path.exists(npy_predict):
    os.makedirs(npy_predict)

csv_predict = os.path.join(base, 'csv_predict')
if not os.path.exists(csv_predict):
    os.makedirs(csv_predict)

model = load_model(filename)

print('Evaluating {}'.format(name))
score = model.evaluate(X_val, Y_val)
print(score)

print('Predicting X_test...')
result = model.predict(X_test)
np.save(os.path.join(npy_predict, 'mow_cnn2d_semi_test_{}.npy'.format(fold_num+1)), result)

df = pd.DataFrame(result)
df.insert(0, 'fname', Y_fname_test)
df.to_csv(os.path.join(csv_predict, 'mow_cnn2d_semi_test_{}.csv'.format(fold_num+1)), index=False, header=True)

