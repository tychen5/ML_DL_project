import os
import sys
import time
import math
import random
import pickle
import numpy as np
import pandas as pd

from keras.utils import to_categorical

with open('./map.pkl', 'rb') as f:
    map_dict = pickle.load(f)

train_label_path = os.path.join('/tmp2/b03902110', 'Y_train_ens_verified.csv')
Y_train = pd.read_csv(train_label_path)

Y_dict = Y_train['label_verified'].map(map_dict)
Y_dict = np.array(Y_dict)

Y_all = []
for i in Y_dict:
    Y_all.append(to_categorical(i, num_classes=41))
Y_all = np.array(Y_all)

print(Y_all)
np.save('/tmp2/b03902110/Y_train_ens_verified.npy', Y_all)
exit()

X_train_path = os.path.join(base_path, 'X_train.npy')
X_all = np.load(X_train_path)

X_all = X_all[verified, :]

idx = list(range(X_all.shape[0]))
random.shuffle(idx)

xSize = math.ceil(X_all.shape[0] / num_fold)

split_X_path = os.path.join(base_path, 'X')
split_y_path = os.path.join(base_path, 'y')
split_fname_path = os.path.join(base_path, 'fname')

if not os.path.exists(split_X_path):
    os.makedirs(split_X_path)
if not os.path.exists(split_y_path):
    os.makedirs(split_y_path)
if not os.path.exists(split_fname_path):
    os.makedirs(split_fname_path)

for i in range(num_fold):
    X = X_all[idx[i*xSize:i*xSize+xSize]]
    y = Y_all[idx[i*xSize:i*xSize+xSize]]
    fname = fname_all[idx[i*xSize:i*xSize+xSize]]
    print('Saving fold {}'.format(i+1))
    print('X.shape:', X.shape)
    print('y.shape:', y.shape)
    print('fname.shape:', fname.shape)
    filename = os.path.join(split_X_path, 'X' + str(i+1) + '.npy')
    np.save(filename, X)
    filename = os.path.join(split_y_path, 'y' + str(i+1) + '.npy')
    np.save(filename, y)
    filename = os.path.join(split_fname_path, 'fname' + str(i+1) + '.npy')
    np.save(filename, fname)
    time.sleep(1)
