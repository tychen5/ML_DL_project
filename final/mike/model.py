import os
import sys
sys.path.append('common')
import util

import keras
from keras.layers import Dense, Conv2D, Convolution2D, MaxPooling2D, GlobalMaxPooling2D, Activation, Dropout, BatchNormalization, Flatten, Input
from keras.models import Model, Sequential

def model_mlt_cnn_alexnet(input_shape, num_classes_array, freeze_mode=0): # AlexNet based
    if freeze_mode > 0:
        print('model_mlt_cnn_alexnet: freeze other than output_%d.' % (freeze_mode))
    conv_trainable = (freeze_mode == 0)
    
    inputs = Input(shape=input_shape)
    layer = Convolution2D(48, 11,  strides=(2,3), activation='relu', padding='same')
    layer.trainable = conv_trainable
    x = layer(inputs)
    layer = MaxPooling2D(3, strides=(1,2))
    layer.trainable = conv_trainable
    x = layer(x)
    layer = BatchNormalization()
    layer.trainable = conv_trainable
    x = layer(x)

    layer = Convolution2D(128, 5, strides=(2,3), activation='relu', padding='same')
    layer.trainable = conv_trainable
    x = layer(x)
    layer = MaxPooling2D(3, strides=2)
    layer.trainable = conv_trainable
    x = layer(x)
    layer = BatchNormalization()
    layer.trainable = conv_trainable
    x = layer(x)

    layer = Convolution2D(192, 3, strides=1, activation='relu', padding='same')
    layer.trainable = conv_trainable
    x = layer(x)
    layer = Convolution2D(192, 3, strides=1, activation='relu', padding='same')
    layer.trainable = conv_trainable
    x = layer(x)
    layer = Convolution2D(128, 3, strides=1, activation='relu', padding='same')
    layer.trainable = conv_trainable
    x = layer(x)
    layer = MaxPooling2D(3, strides=(1,2))
    layer.trainable = conv_trainable
    x = layer(x)
    layer = BatchNormalization()
    layer.trainable = conv_trainable
    x = layer(x)

    ys = []
    for i, num_classes in enumerate(num_classes_array):
        trainable = (freeze_mode == i + 1)
        fc_size = 256
        layer = Flatten()
        layer.trainable = trainable
        y = layer(x)
        layer = Dense(fc_size, activation='relu')
        layer.trainable = trainable
        y = layer(y)
        layer = Dropout(0.5)
        layer.trainable = trainable
        y = layer(y)
        layer = Dense(fc_size, activation='relu')
        layer.trainable = trainable
        y = layer(y)
        layer = Dropout(0.5)
        layer.trainable = trainable
        y = layer(y)
        layer = Dense(num_classes, activation='softmax', name='output_%d' % (i + 1))
        layer.trainable = trainable
        y = layer(y)
        ys.append(y)

    model = Model(inputs=inputs, outputs=ys)
    return model

def model_cnn_alexnet(input_shape, num_classes): # AlexNet based, for single dataset
    model = Sequential()
 
    model.add(Conv2D(48, 11,  input_shape=input_shape, strides=(2,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(3, strides=(1,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, 5, strides=(2,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(3, strides=2))
    model.add(BatchNormalization())

    model.add(Conv2D(192, 3, strides=1, activation='relu', padding='same'))
    model.add(Conv2D(192, 3, strides=1, activation='relu', padding='same'))
    model.add(Conv2D(128, 3, strides=1, activation='relu', padding='same'))
    model.add(MaxPooling2D(3, strides=(1,2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model
