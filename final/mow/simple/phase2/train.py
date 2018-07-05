import os
import sys
import numpy as np

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint

def getTrainData():
    X = []
    y = []

    for i in range(6):
        fileX = '/tmp2/b03902110/phase2/data/X/X' + str(i+1) + '.npy'
        fileY = '/tmp2/b03902110/phase2/data/y/y' + str(i+1) + '.npy'
        
        X.append(np.load(fileX))
        y.append(np.load(fileY))

    X = np.array(X)
    y = np.array(y)

    return X, y

def split_data(X, y, idx):
    X_train = []
    y_train = []
    
    for i in range(6):
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
    X = np.concatenate((X_train, X_val))

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return X_train, X_val

if __name__ == '__main__':
    val_set_num = str(sys.argv[1])
    X, y = getTrainData()
    X_train, y_train, X_val, y_val = split_data(X, y, int(val_set_num))
    X_train, X_val = normalize(X_train, X_val)

    model = Sequential()

    model.add(Convolution2D(64, (4, 23), input_shape=(40, 345, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Convolution2D(64, (4, 5)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(128, (2, 23)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Convolution2D(128, (2, 5)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(41))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    filename = '/tmp2/b03902110/phase2/full_6fold_model/model' + val_set_num
    if not os.path.exists(filename):
        os.makedirs(filename)

    callback = ModelCheckpoint(filename + '/weights.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.h5', monitor='val_loss', save_best_only=False, period=1)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=128, epochs=50, callbacks=[callback])

    print('\n\n========== Done ==========\n\n')
    
    score = model.evaluate(X_val, y_val)
    print(score)

    model.save('/tmp2/b03902110/phase2/6_fold_model/cnn_model' + val_set_num)
