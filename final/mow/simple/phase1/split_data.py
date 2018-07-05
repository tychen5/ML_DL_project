import random
import pickle
import numpy as np
import pandas as pd

from keras.utils import to_categorical

with open('map.pkl', 'rb') as f:
    map_dict = pickle.load(f)

Y_train = pd.read_csv('./data/train/train3_Y.csv')
Y_dict = Y_train['label'].map(map_dict)
Y_dict = np.array(Y_dict)
print(Y_dict.shape)

Y_all = []
for i in Y_dict:
    Y_all.append(to_categorical(i, num_classes=41))
Y_all = np.array(Y_all)
print(Y_all)
print(Y_all.shape)

X_all = np.load('./data/train/train3_X.npy')

idx = list(range(X_all.shape[0]))
random.shuffle(idx)

xSize = int(X_all.shape[0] / 6)

for i in range(6):
    X = X_all[idx[i*xSize:i*xSize+xSize]]
    y = Y_all[idx[i*xSize:i*xSize+xSize]]
    print(X.shape)
    print(y.shape)
    filename = './data/X/X' + str(i+1) + '.npy'
    np.save(filename, X)
    filename = './data/y/y' + str(i+1) + '.npy'
    np.save(filename, y)

