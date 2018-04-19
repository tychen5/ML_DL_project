
# coding: utf-8

# In[1]:


try:
    import pandas as pd
    import tensorflow as tf
    import os, sys
    import numpy as np
    from random import shuffle
    from math import log, floor
    from keras import backend as K
    from keras.models import *
    from keras.layers import *
    from keras.activations import *
    from keras.callbacks import *
    from keras.utils import *
    from keras.layers.advanced_activations import *
    from keras import *
    from keras.engine.topology import *
    from keras.optimizers import *
    import keras
    import pickle
    from keras.applications import *
    from keras.preprocessing.image import *
    K.set_image_dim_ordering('tf')
    from utils import feature, utils
except:
    print('import error')
    pass


# In[2]:


def trans_data(X_train): #feature    
    X_T=[]
    for i,v in enumerate(X_train):
        X_T.append(np.fromstring(v, dtype=float, sep=' ').reshape((48, 48, 1))*(1/255))
    return np.array(X_T)


# In[3]:


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
    X_all = X_all.reshape(-1,2304)
    X_test = X_test.reshape(-1,2304)
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / (sigma+1e-20)

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]].reshape(-1,48, 48, 1)
    X_test = X_train_test_normed[X_all.shape[0]:].reshape(-1,48, 48, 1)
    return X_all, X_test


# In[9]:


# X_train = pd.read_csv('data/train.csv', sep=',', header=0) ##路徑注意
X_train = pd.read_pickle('models/norm')
X_test = pd.read_csv(sys.argv[1], sep=',', header=0) ##路徑更改
Y_train = X_train['label'].astype(np.int)
X_train = X_train['feature']
X_test = X_test['feature']
X_train = trans_data(X_train)
Y_train = trans_label(Y_train)
X_test = trans_data(X_test)
X_train, X_test = normalize(X_train, X_test)
print(X_train.shape,Y_train.shape,X_test.shape)


# In[8]:


# X_train = pd.read_pickle('models/norm')
# X_train.to_pickle('models/norm')


# In[6]:


#model_normal
#KUAN_FS_0.73
model = Sequential()
model.add(Conv2D(256, (4,4), padding='same',kernel_initializer='glorot_normal', input_shape=(48,48,1))) #same??
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Dropout(0.25))
model.add(Conv2D(512, (4,4),kernel_initializer='glorot_normal',padding='same'))
model.add(LeakyReLU(alpha=0.05))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Dropout(0.35))
model.add(Conv2D(1024, (4,4),kernel_initializer='glorot_normal',padding='same'))
model.add(LeakyReLU(alpha=0.05))  #softplus
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Dropout(0.4))
model.add(Conv2D(2048, (3,3),kernel_initializer='glorot_normal',padding='same')) #5,5?
model.add(LeakyReLU(alpha=0.05))  #softplus
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.45))
model.add(Flatten())
model.add(Dense(1024,activation='softplus',kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.02))) #2048,selu,l2
model.add(BatchNormalization())
model.add(Dropout(0.75))
model.add(Dense(1024,kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.01))) #2048,selu,l2
model.add(BatchNormalization())
model.add(Activation('softplus'))
model.add(Dropout(0.8))
model.add(Dense(7))
model.add(Activation('softmax'))
model.summary()


# In[7]:


#model2_normal
#KUAN_FS_0.7
model2 = Sequential()
model2.add(Conv2D(64, (4,4), padding='same', input_shape=(48,48,1),kernel_initializer='glorot_normal')) #he_normal
model2.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model2.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
model2.add(BatchNormalization())
model2.add(Dropout(0.25))
model2.add(Conv2D(64, (4,4), padding='same',kernel_initializer='glorot_normal')) #same??
model2.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model2.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model2.add(Dropout(0.25))
model2.add(Conv2D(128, (4,4),kernel_initializer='glorot_normal',padding='same'))
model2.add(LeakyReLU(alpha=0.05))
model2.add(BatchNormalization())
model2.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model2.add(Dropout(0.35))
model2.add(Conv2D(128, (4,4),kernel_initializer='glorot_normal',padding='same'))
model2.add(LeakyReLU(alpha=0.05))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model2.add(Dropout(0.35))
model2.add(Conv2D(512, (3,3),kernel_initializer='glorot_normal',padding='same'))
model2.add(LeakyReLU(alpha=0.05))  #softplus
model2.add(BatchNormalization())
model2.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model2.add(Dropout(0.4))
model2.add(Conv2D(512, (3,3),kernel_initializer='glorot_normal',padding='same')) #5,5?
model2.add(LeakyReLU(alpha=0.05))  #softplus
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model2.add(Dropout(0.4))
model2.add(Flatten())
model2.add(Dense(512,kernel_initializer='glorot_normal',activation='softplus')) #2048,selu,l2
model2.add(BatchNormalization())
model2.add(Dropout(0.5))
model2.add(Dense(7))
model2.add(Activation('softmax'))
model2.summary()


# In[8]:


model.load_weights('models/cnn_normal_0.7_3.h5_weights.h5') ## 路徑注意
ans = model.predict(X_test)
model2.load_weights('models/cnn_normal_0.7.h5_weights.h5')  ## 路徑注意
ans2 = model2.predict(X_test)


# In[9]:


prediction = ((ans ** 0.68626) * (ans2 ** 0.66648))** (1/1.35274) #073+07_square_p1
# p1 = ((ans ** 0.68626) * (ans2 ** 0.66648))** (1/1.35274) #plus2
# p2 = (ans*0.68626 + ans2*0.66648)/1.35274 #plus2
# prediction = (p1+p2) #plus2


# In[ ]:


kk = np.argmax(prediction,axis=1)
with open(sys.argv[2] , 'w') as f:  #路徑改掉
    f.write('id,label\n')
    for i, v in  enumerate(kk):
        f.write('%d,%d\n' %(i, v))

