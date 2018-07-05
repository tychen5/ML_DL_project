import os
import sys
import numpy as np

from keras import regularizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping

import resnet

base_path = '/home/tyt/how2ml/mfcc4'
base_data_path = os.path.join(base_path, 'data')
num_fold = 10

def getTrainData():
    X = []
    y = []

    for i in range(num_fold):
        fileX = os.path.join(base_data_path, 'X/X' + str(i+1) + '.npy')
        fileY = os.path.join(base_data_path, 'y/y' + str(i+1) + '.npy')
        
        X.append(np.load(fileX))
        y.append(np.load(fileY))

    X = np.array(X)
    y = np.array(y)

    return X, y

def split_data(X, y, idx):
    X_train = []
    y_train = []
    
    for i in range(num_fold):
        if i == idx:
            X_val = X[i]
            y_val = y[i]
            continue
        if X_train == []:
            X_train = X[i]
            y_train = y[i]
        else:
            X_train = np.concatenate((X_train, X[i]))
            y_train = np.concatenate((y_train, y[i]))

    return X_train, y_train, X_val, y_val

def normalize(X_train, X_val):
    filename = os.path.join(base_data_path, 'X_train.npy')
    X = np.load(filename)

    filename = os.path.join(base_data_path, 'X_test.npy')
    X_test = np.load(filename)

    X = np.concatenate((X, X_test))

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return X_train, X_val

if __name__ == '__main__':
	for i in range(8, 10):
		print('Fold {}'.format(i+1))
		val_set_num = str(i)
		X, y = getTrainData()
		X_train, y_train, X_val, y_val = split_data(X, y, int(val_set_num))
		# X_train, X_val = normalize(X_train, X_val)

		filename = os.path.join(base_path, 'full_10fold_model/model' + val_set_num)
		if not os.path.exists(filename):
			os.makedirs(filename)

		callback = ModelCheckpoint(filename + '/weights.{epoch:04d}-{val_loss:.4f}-{val_acc:.4f}.h5', monitor='val_acc', save_best_only=True)
		early = EarlyStopping(monitor='val_acc', mode='max', patience=80)

		model = resnet.ResnetBuilder.build_resnet_152((1, 20, 690), 41)

		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.summary()

		model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=10000, callbacks=[callback, early])

		print('\n\n========== Done ==========\n\n')
	
