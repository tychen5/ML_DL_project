
# coding: utf-8

# # Confusion Matrix
# 
# 0.73 model

# In[1]:


from keras.models import load_model
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os, pickle, itertools
from termcolor import colored, cprint
from PIL import Image
from keras import backend as K
from utils import feature, utils
import os, sys
import numpy as np
from random import shuffle
from math import log, floor
import pandas as pd
import tensorflow as tf
import tensorboard as tb
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.activations import *
from keras.callbacks import *
from keras.utils import *
from keras.layers.advanced_activations import *
from keras.layers.advanced_activations import *
from keras import *
from keras.engine.topology import *
from keras.optimizers import *
import keras
import pandas as pd
import numpy as np
# import sklearn
import pickle
from keras.applications import *
from keras.preprocessing.image import *
K.set_image_dim_ordering('tf')

from utils import feature, utils

K.set_learning_phase(0)

#model_normal
#KUAN_FS_0.73
model = Sequential()
# model.add(Conv2D(64, (3,3), padding='same', input_shape=(48,48,1),kernel_initializer='glorot_normal')) #he_normal
# model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
# model.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
# model.add(BatchNormalization())
# model.add(Dropout(0.25))

model.add(Conv2D(256, (4,4), padding='same',kernel_initializer='glorot_normal', input_shape=(48,48,1))) #same??
# model.add(LeakyReLU(alpha=0.05))
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
# model.add(Dropout(0.25))
model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Dropout(0.25))



# model.add(Conv2D(128, (4,4),kernel_initializer='glorot_normal',padding='same'))
# model.add(LeakyReLU(alpha=0.05))
# model.add(BatchNormalization())
# model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
# model.add(Dropout(0.35))
# model.add(Activation('relu'))

model.add(Conv2D(512, (4,4),kernel_initializer='glorot_normal',padding='same'))
model.add(LeakyReLU(alpha=0.05))
model.add(BatchNormalization())
# model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Dropout(0.35))



model.add(Conv2D(1024, (4,4),kernel_initializer='glorot_normal',padding='same'))
model.add(LeakyReLU(alpha=0.05))  #softplus
# model.add(Activation('softplus'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Dropout(0.4))
# model.add(BatchNormalization())
# model.add(Activation('relu'))

model.add(Conv2D(2048, (3,3),kernel_initializer='glorot_normal',padding='same')) #5,5?
model.add(LeakyReLU(alpha=0.05))  #softplus
# model.add(Activation('softplus'))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
# model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Dropout(0.45))
# model.add(BatchNormalization())

# model.add(Conv2D(256, (3,3),kernel_initializer='glorot_normal',padding='same'))
# # model.add(Activation('softplus'))
# model.add(LeakyReLU(alpha=0.05))
# model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Conv2D(256, (3,3),kernel_initializer='glorot_normal',padding='same'))
# # model.add(Activation('softplus'))
# model.add(LeakyReLU(alpha=0.05))
# model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
# model.add(BatchNormalization())
# # model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))


model.add(Flatten())



# model.add(Dense(2048,kernel_initializer='lecun_normal',kernel_regularizer=regularizers.l2(0.011))) #0.02
# model.add(BatchNormalization())
# model.add(Activation('selu'))
# model.add(Dropout(0.5))

# model.add(Dense(2048,kernel_initializer='lecun_normal',kernel_regularizer=regularizers.l2(0.011))) #0.02=> 512 *2
# model.add(BatchNormalization())
# model.add(Activation('selu'))
# model.add(Dropout(0.5))

# model.add(Dense(512,kernel_initializer='lecun_normal',kernel_regularizer=regularizers.l2(0.001))) #NEW
# model.add(BatchNormalization())
# model.add(Activation('selu'))
# model.add(Dropout(0.5))

# model.add(Dense(512,kernel_initializer='lecun_normal',kernel_regularizer=regularizers.l2(0.001))) #NEW
# model.add(BatchNormalization())
# model.add(Activation('selu'))
# model.add(Dropout(0.5))

model.add(Dense(1024,activation='softplus',kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.02))) #2048,selu,l2
model.add(BatchNormalization())
# model.add(Activation('softplus'))
model.add(Dropout(0.75))

model.add(Dense(1024,kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.01))) #2048,selu,l2
model.add(BatchNormalization())
model.add(Activation('softplus'))
model.add(Dropout(0.8))
# model.add(Dense(128,kernel_initializer='glorot_normal',activation='elu',kernel_regularizer=regularizers.l2(0.0001))) #2048,selu,l2
# model.add(BatchNormalization())
# model.add(Activation('softplus'))
# model.add(Dropout(0.5))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(32))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(7))
model.add(Activation('softmax'))
# model.summary()
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

def main2():
    model_name = "cnn_normal_0.7_3.h5"
    model_path = os.path.join('model/', model_name)
    model.load_weights(model_path)
    print(colored("Loaded model from {}".format(model_name), 'blue', attrs=['bold'])) #yellow

    with open('data/valid_73.pickle', 'rb') as handle:
        X_valid,Y_valid = pickle.load(handle)
    X_valid = X_valid.reshape((2871,2304))
    private_pixels = [ X_valid[i].reshape((1, 48, 48, 1)) for i in range(len(X_valid)) ]
#     print(X_valid.shape)
#     private_pixels = [ X_valid.astype('int64')]
    
    input_img = model.input
    img_ids = [i for i in range(5)]

    for idx in img_ids:
        val_proba = model.predict(private_pixels[idx])
        pred = val_proba.argmax(axis=1) #-1
        target = K.mean(model.output[:, pred])
        grads = K.gradients(target, input_img)[0] * 255.0
        fn = K.function([input_img, K.learning_phase()], [grads])
        
        # val_grads = fn([private_pixels[idx].reshape(-1, 48, 48, 1), 0])[0].reshape(48, 48, -1)
        # val_grads *= -1
        # val_grads = np.max(np.abs(val_grads), axis=-1, keepdims=True)
        # heatmap = val_grads.reshape(48, 48)
        
        heatmap = fn([private_pixels[idx], 0])
        heatmap = heatmap[0].reshape(48, 48)
        
        thres = 0.5
        see = private_pixels[idx].reshape(48, 48)
       
        # Plot original image
        plt.figure()
        plt.imshow(see*255.0,cmap='gray')
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(ori_dir, 'original_{}.png'.format(idx)), dpi=100)

        see[np.where(heatmap <= thres)] = np.mean(see)
       
        # Plot saliency heatmap
        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(cmap_dir, 'heatmap_{}.png'.format(idx)), dpi=100)

        # Plot "Little-heat-part masked"
        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(partial_see_dir, '{}.png'.format(idx)), dpi=100)

# if __name__ == "__main__":

base_dir = os.path.join('./result')
img_dir = os.path.join(base_dir, 'image')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
    os.makedirs(cmap_dir)
partial_see_dir = os.path.join(img_dir,'partial_see')
if not os.path.exists(partial_see_dir):
    os.makedirs(partial_see_dir)
ori_dir = os.path.join(img_dir, 'original')
if not os.path.exists(ori_dir):
    os.makedirs(ori_dir)

main2()

