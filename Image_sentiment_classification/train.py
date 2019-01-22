
# coding: utf-8

# In[1]:


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
# from keras.layers.advanced_activations import *
from keras import *
from keras.engine.topology import *
from keras.optimizers import *
import keras
# import pandas as pd
# import numpy as np
# import sklearn
import pickle
from keras.applications import *
from keras.preprocessing.image import *
K.set_image_dim_ordering('tf')

from utils import feature, utils


# In[2]:



def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid


# In[3]:


def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
#     print(X.shape, Y.shape)
    return (X[randomize], Y[randomize])


# In[4]:


def trans_data(X_train): #feature    
    X_T=[]
    for i,v in enumerate(X_train):
        X_T.append(np.fromstring(v, dtype=float, sep=' ').reshape((48, 48, 1))*(1/255))
    return np.array(X_T)


# In[5]:


def trans_label(X_label): #label   
    Y_T=[]
    for i,v in enumerate(X_label):
        onehot = np.zeros((7, ), dtype=np.float)
        onehot[v] = 1.
        Y_T.append(onehot)
    return np.array(Y_T)


# In[6]:


def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_all = X_all.reshape((-1,2304))
    X_test = X_test.reshape((-1,2304))
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / (sigma+1e-20)

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]].reshape((-1,48, 48, 1))
    X_test = X_train_test_normed[X_all.shape[0]:].reshape((-1,48, 48, 1))
    return X_all, X_test


# In[7]:


X_train = pd.read_csv(sys.argv[1], sep=',', header=0)  ##路徑更改
X_test = pd.read_csv('data/test.csv', sep=',', header=0) ## 路徑注意
Y_train = X_train['label'].astype(np.int)
X_train = X_train['feature']
X_test = X_test['feature']
X_train = trans_data(X_train)
Y_train = trans_label(Y_train)
X_test = trans_data(X_test)
X_train, X_test = normalize(X_train, X_test)
X_train, Y_train, X_valid, Y_valid = split_valid_set(X_train, Y_train, 0.9)
print(X_train.shape,Y_train.shape,X_test.shape)


# In[8]:


class History(Callback):
    def on_train_begin(self,logs={}):
        self.tr_losses=[]
        self.val_losses=[]
        self.tr_accs=[]
        self.val_accs=[]

    def on_epoch_end(self,epoch,logs={}):
        self.tr_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.tr_accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))


# In[9]:


datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
#     samplewise_center=True,
#     samplewise_std_normalization=True,
    rotation_range=20, #40 20
    width_shift_range=0.2, #0.5
    height_shift_range=0.2, #0.5  
    horizontal_flip=True,
    shear_range=0.001,
#     zoom_ranqge=0.001,
    data_format='channels_last'

) #zoom_range=0.2
datagen.fit(X_train)


# In[10]:


#model_normal
#KUAN_FS_0.7
model = Sequential()
model.add(Conv2D(64, (4,4), padding='same', input_shape=(48,48,1),kernel_initializer='glorot_normal')) #he_normal
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(64, (4,4), padding='same',kernel_initializer='glorot_normal')) #same??
# model.add(LeakyReLU(alpha=0.05))
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
# model.add(Dropout(0.25))
model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Dropout(0.25))



model.add(Conv2D(128, (4,4),kernel_initializer='glorot_normal',padding='same'))
model.add(LeakyReLU(alpha=0.05))
model.add(BatchNormalization())
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Dropout(0.35))
# model.add(Activation('relu'))

model.add(Conv2D(128, (4,4),kernel_initializer='glorot_normal',padding='same'))
model.add(LeakyReLU(alpha=0.05))
model.add(BatchNormalization())
# model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Dropout(0.35))



model.add(Conv2D(512, (3,3),kernel_initializer='glorot_normal',padding='same'))
model.add(LeakyReLU(alpha=0.05))  #softplus
model.add(BatchNormalization())
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Dropout(0.4))
# model.add(BatchNormalization())
# model.add(Activation('relu'))

model.add(Conv2D(512, (3,3),kernel_initializer='glorot_normal',padding='same')) #5,5?
model.add(LeakyReLU(alpha=0.05))  #softplus
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Dropout(0.4))
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

model.add(Dense(512,kernel_initializer='glorot_normal',activation='softplus')) #2048,selu,l2
model.add(BatchNormalization())
# model.add(Activation('softplus'))
model.add(Dropout(0.5))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(32))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(7))
model.add(Activation('softmax'))
model.summary()
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[12]:


#model2_normal
#KUAN_FS_0.73
model2 = Sequential()
# model2.add(Conv2D(64, (3,3), padding='same', input_shape=(48,48,1),kernel_initializer='glorot_normal')) #he_normal
# model2.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
# model2.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
# model2.add(BatchNormalization())
# model2.add(Dropout(0.25))

model2.add(Conv2D(256, (4,4), padding='same',kernel_initializer='glorot_normal', input_shape=(48,48,1))) #same??
# model2.add(LeakyReLU(alpha=0.05))
model2.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model2.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
# model2.add(Dropout(0.25))
model2.add(BatchNormalization())
# model2.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model2.add(Dropout(0.25))



# model2.add(Conv2D(128, (4,4),kernel_initializer='glorot_normal',padding='same'))
# model2.add(LeakyReLU(alpha=0.05))
# model2.add(BatchNormalization())
# model2.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
# model2.add(Dropout(0.35))
# model2.add(Activation('relu'))

model2.add(Conv2D(512, (4,4),kernel_initializer='glorot_normal',padding='same'))
model2.add(LeakyReLU(alpha=0.05))
model2.add(BatchNormalization())
# model2.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model2.add(Dropout(0.35))



model2.add(Conv2D(1024, (4,4),kernel_initializer='glorot_normal',padding='same'))
model2.add(LeakyReLU(alpha=0.05))  #softplus
# model2.add(Activation('softplus'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model2.add(Dropout(0.4))
# model2.add(BatchNormalization())
# model2.add(Activation('relu'))

model2.add(Conv2D(2048, (3,3),kernel_initializer='glorot_normal',padding='same')) #5,5?
model2.add(LeakyReLU(alpha=0.05))  #softplus
# model2.add(Activation('softplus'))
model2.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2)))
# model2.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
# model2.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model2.add(Dropout(0.45))
# model2.add(BatchNormalization())

# model2.add(Conv2D(256, (3,3),kernel_initializer='glorot_normal',padding='same'))
# # model2.add(Activation('softplus'))
# model2.add(LeakyReLU(alpha=0.05))
# model2.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
# model2.add(BatchNormalization())
# model2.add(Dropout(0.5))

# model2.add(Conv2D(256, (3,3),kernel_initializer='glorot_normal',padding='same'))
# # model2.add(Activation('softplus'))
# model2.add(LeakyReLU(alpha=0.05))
# model2.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
# model2.add(BatchNormalization())
# # model2.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
# model2.add(MaxPooling2D(pool_size=(2, 2)))
# model2.add(Dropout(0.5))


model2.add(Flatten())



# model2.add(Dense(2048,kernel_initializer='lecun_normal',kernel_regularizer=regularizers.l2(0.011))) #0.02
# model2.add(BatchNormalization())
# model2.add(Activation('selu'))
# model2.add(Dropout(0.5))

# model2.add(Dense(2048,kernel_initializer='lecun_normal',kernel_regularizer=regularizers.l2(0.011))) #0.02=> 512 *2
# model2.add(BatchNormalization())
# model2.add(Activation('selu'))
# model2.add(Dropout(0.5))

# model2.add(Dense(512,kernel_initializer='lecun_normal',kernel_regularizer=regularizers.l2(0.001))) #NEW
# model2.add(BatchNormalization())
# model2.add(Activation('selu'))
# model2.add(Dropout(0.5))

# model2.add(Dense(512,kernel_initializer='lecun_normal',kernel_regularizer=regularizers.l2(0.001))) #NEW
# model2.add(BatchNormalization())
# model2.add(Activation('selu'))
# model2.add(Dropout(0.5))

model2.add(Dense(1024,activation='softplus',kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.02))) #2048,selu,l2
model2.add(BatchNormalization())
# model2.add(Activation('softplus'))
model2.add(Dropout(0.75))

model2.add(Dense(1024,kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.01))) #2048,selu,l2
model2.add(BatchNormalization())
model2.add(Activation('softplus'))
model2.add(Dropout(0.8))
# model2.add(Dense(128,kernel_initializer='glorot_normal',activation='elu',kernel_regularizer=regularizers.l2(0.0001))) #2048,selu,l2
# model2.add(BatchNormalization())
# model2.add(Activation('softplus'))
# model2.add(Dropout(0.5))
# model2.add(Dense(64))
# model2.add(Activation('relu'))
# model2.add(Dropout(0.5))
# model2.add(Dense(32))
# model2.add(Activation('relu'))
# model2.add(Dropout(0.5))
model2.add(Dense(7))
model2.add(Activation('softmax'))
model2.summary()
# model2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


opt2 = Adam(lr=0.02,decay=1e-6) #lr=0.015,decay=1e-8 ,amsgrad=True
opt1 = Adam(decay=1e-8)
model.compile(optimizer=opt1, loss='categorical_crossentropy', metrics=['accuracy'])
model2.compile(optimizer=opt2, loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


batchSize=128
patien=100
epoch=1500
saveP = 'model/cnn_normal_0.7.h5'
logD = './logs/'
history = History()
callback=[
    EarlyStopping(patience=patien,monitor='val_loss',verbose=1),
    ModelCheckpoint(saveP,monitor='val_acc',verbose=1,save_best_only=True, save_weights_only=False),
    TensorBoard(log_dir=logD+'events.epochs'+str(epoch)),
    history#,batch_size=batch_size, write_graph=True, write_grads=False, write_images=True)
]
model.fit_generator(datagen.flow(X_train,Y_train,batch_size=batchSize),steps_per_epoch=int(len(X_train)/batchSize),epochs=epoch,
                    validation_data=(X_valid,Y_valid),callbacks=callback, class_weight='auto')
utils.dump_history(logD,history)
model.save_weights(saveP+"_weights.h5")
#loss: 0.7588 - acc: 0.7179 - val_loss: 0.9542 - val_acc: 0.6809


# In[ ]:


batchSize=128
patien=100
epoch=1500
saveP = 'model/cnn_normal_0.7_3.h5'
logD = './logs/'
history = History()
callback=[
    EarlyStopping(patience=patien,monitor='val_acc',verbose=1),
    ModelCheckpoint(saveP,monitor='val_loss',verbose=1,save_best_only=True, save_weights_only=False),
    TensorBoard(log_dir=logD+'events.epochs'+str(epoch)),
    history#,batch_size=batch_size, write_graph=True, write_grads=False, write_images=True)
]
model2.fit_generator(datagen.flow(X_train,Y_train,batch_size=batchSize),steps_per_epoch=int(len(X_train)/batchSize),epochs=epoch,
                    validation_data=(X_valid,Y_valid),callbacks=callback, class_weight='auto')
utils.dump_history(logD,history)
model.save_weights(saveP+"_weights.h5")
#0.6832

