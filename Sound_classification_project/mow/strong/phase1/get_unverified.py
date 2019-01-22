import os
import sys
import time
import math
import random
import pickle
import numpy as np
import pandas as pd

from keras.utils import to_categorical

# map_path = path of 'map.pkl'
map_path = sys.argv[1]

# base_path = directory of all the data (train_label.csv, X_train.npy)
base_path = sys.argv[2]

# num_fold
num_fold = int(sys.argv[3])

with open(map_path, 'rb') as f:
    map_dict = pickle.load(f)

verified = []
unverified = []

train_label_path = os.path.join(base_path, 'train_label.csv')
Y_train = pd.read_csv(train_label_path)

for i in range(len(Y_train)):
    if Y_train['manually_verified'][i] == 1:
        verified.append(i)
    else:
        unverified.append(i)

Y_un = Y_train.loc[unverified,:]

fname_all = Y_un['fname']
fname_all = np.array(fname_all)

Y_dict = Y_un['label'].map(map_dict)
Y_dict = np.array(Y_dict)

Y_all = []
for i in Y_dict:
    Y_all.append(to_categorical(i, num_classes=41))
Y_all = np.array(Y_all)

filename = os.path.join(base_path, 'fname_unverified.npy')
np.save(filename, fname_all)

filename = os.path.join(base_path, 'y_unverified.npy')
np.save(filename, Y_all)

X_train_path = os.path.join(base_path, 'X_train.npy')
X_all = np.load(X_train_path)

X_un = X_all[unverified, :]
filename = os.path.join(base_path, 'X_unverified.npy')
np.save(filename, X_un)

'''
exit()

X_all = X_all[verified, :]

idx = list(range(X_all.shape[0]))
random.shuffle(idx)

xSize = math.ceil(X_all.shape[0] / num_fold)

split_X_path = os.path.join(base_path, 'X')
split_y_path = os.path.join(base_path, 'y')

if not os.path.exists(split_X_path):
    os.makedirs(split_X_path)
if not os.path.exists(split_y_path):
    os.makedirs(split_y_path)


for i in range(num_fold):
    X = X_all[idx[i*xSize:i*xSize+xSize]]
    y = Y_all[idx[i*xSize:i*xSize+xSize]]
    print('Saving fold {}'.format(i+1))
    print('X.shape:', X.shape)
    print('y.shape:', y.shape)
    filename = os.path.join(split_X_path, 'X' + str(i+1) + '.npy')
    np.save(filename, X)
    filename = os.path.join(split_y_path, 'y' + str(i+1) + '.npy')
    np.save(filename, y)
    time.sleep(1)
'''
