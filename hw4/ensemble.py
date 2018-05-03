
# coding: utf-8

# In[1]:


import os, sys
import numpy as np
from random import shuffle
import argparse
from math import log, floor
import pandas as pd
import tensorflow as tf
import tensorboard as tb
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.activations import *
from keras.callbacks import *
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.utils import *
from keras.layers.advanced_activations import *
from keras.layers.advanced_activations import *
from keras import *
from keras.engine.topology import *
from keras.optimizers import *
import keras
import pandas as pd
import numpy as np
import scipy as sp
import sklearn
from sklearn.model_selection import KFold
import pickle


# In[2]:


def load_data(train_data_path, train_label_path, test_data_path,cols=None):
    if cols == None:
        X_train = pd.read_csv(train_data_path, sep=',', header=0)
        X_test = pd.read_csv(test_data_path, sep=',', header=0)
    else:
        X_train = pd.read_csv(train_data_path, sep=',')
        X_train['divide'] = X_train['capital_gain'].astype(float)/X_train['hours_per_week'].astype(float)
        X_train['minus'] = X_train['capital_gain']-X_train['capital_loss']
        X_train = X_train.filter(items=cols)
        
        X_test = pd.read_csv(test_data_path, sep=',')
        X_test['divide'] = X_test['capital_gain'].astype(float)/X_test['hours_per_week'].astype(float)
        X_test['minus'] = X_test['capital_gain']-X_test['capital_loss']
        X_test = X_test.filter(items=cols)
    X_train = np.array(X_train.values)
    Y_train = pd.read_csv(train_label_path, sep=',', header=0)
    Y_train = np.array(Y_train.values)
    X_test = np.array(X_test.values)
#     print(X_train.shape,Y_train.shape)
    return (X_train, Y_train, X_test)


# In[3]:


def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
#     print(X.shape, Y.shape)
    return (X[randomize], Y[randomize])


# In[4]:


def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test


# In[5]:


def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid


# In[6]:


def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))


# In[7]:


def valid(w, b, X_valid, Y_valid):
    valid_data_size = len(X_valid)

    z = (np.dot(X_valid, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    print('Validation acc = %f' % (float(result.sum()) / valid_data_size))
    return


# In[8]:


def train(X_all, Y_all, save_dir):
    # Split a 10%-validation set from the training set
    valid_set_percentage = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)

    # Initiallize parameter, hyperparameter
    w = np.zeros((123,)) #106
    b = np.zeros((1,))
    l_rate = 0.1
    batch_size = 32
    train_data_size = len(X_train)
    step_num = int(floor(train_data_size / batch_size))
    epoch_num = 3000
    save_param_iter = 50

    # Start training
    total_loss = 0.0
    for epoch in range(1, epoch_num):
        # Do validation and parameter saving
        if (epoch) % save_param_iter == 0:
            print('=====Saving Param at epoch %d=====' % epoch)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            np.savetxt(os.path.join(save_dir, 'w'), w)
            np.savetxt(os.path.join(save_dir, 'b'), [b,])
            print('epoch avg loss = %f' % (total_loss / (float(save_param_iter) * train_data_size)))
            total_loss = 0.0
            valid(w, b, X_valid, Y_valid)

        # Random shuffle
        X_train, Y_train = _shuffle(X_train, Y_train)

        # Train with batch
        for idx in range(step_num):
            X = X_train[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]

            z = np.dot(X, np.transpose(w)) + b
            y = sigmoid(z)

            cross_entropy = -1 * (np.dot(np.squeeze(Y), np.log(y)) + np.dot((1 - np.squeeze(Y)), np.log(1 - y)))
            total_loss += cross_entropy

            w_grad = np.mean(-1 * X * (np.squeeze(Y) - y).reshape((batch_size,1)), axis=0)
            b_grad = np.mean(-1 * (np.squeeze(Y) - y))

            # SGD updating parameters
            w = w - l_rate * w_grad
            b = b - l_rate * b_grad

    return


# In[9]:


import numpy as np
from keras.callbacks import Callback
class GetBest(Callback):
    """Get the best model at the end of training.
	# Arguments
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        mode: one of {auto, min, max}.
            The decision
            to overwrite the current stored weights is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
	# Example
		callbacks = [GetBest(monitor='val_acc', verbose=1, mode='max')]
		mode.fit(X, y, validation_data=(X_eval, Y_eval),
                 callbacks=callbacks)
    """

    def __init__(self, monitor='val_loss', verbose=0,
                 mode='auto', period=1):
        super(GetBest, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.best_epochs = 0
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('GetBest mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
                
    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            #filepath = self.filepath.format(epoch=epoch + 1, **logs)
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can pick best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' storing weights.'
                              % (epoch + 1, self.monitor, self.best,
                                 current))
                    self.best = current
                    self.best_epochs = epoch + 1
                    self.best_weights = self.model.get_weights()
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve' %
                              (epoch + 1, self.monitor))            
                    
    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print('Using epoch %05d with %s: %0.5f' % (self.best_epochs, self.monitor,
                                                       self.best))
        self.model.set_weights(self.best_weights)


# In[10]:


##STACKING##
def NN_model(X_T,y_T,x_valid,y_valid):    
    model = Sequential()
    '''
    model.add(Conv1D(16,2,padding='same',input_shape=(1,658)))
    model.add(Conv1D(32,2,padding='same'))
    model.add(Conv1D(32,2,padding='same'))
    model.add(Conv1D(64,2,padding='same'))
    model.add(Conv1D(64,2,padding='same'))  #AF?
    model.add(Flatten())
    '''
    #dropout 0.001 0.5 decay
    # l2: 0.0001~0.01 decay
#     model.add(Dense(8192,kernel_initializer='lecun_normal',input_dim=123,kernel_regularizer=regularizers.l2(0.00001))) #, kernel_regularizer=regularizers.l2(0.0001)
#     model.add(BatchNormalization())
#     model.add(Activation('selu'))
    # model.add(PReLU(alpha_initializer='zero', weights=None))
#     model.add(Dropout(0.2))
    # model.add(BatchNormalization())
    # model.add(PReLU(alpha_initializer='zero'))
#     model.add(Dense(4096,kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.00001)))
#     model.add(BatchNormalization())
    # model.add(PReLU(alpha_initializer='zero', weights=None))
#     model.add(Activation('selu'))
#     model.add(Dropout(0.2))
    # model.add(BatchNormalization())
#     model.add(Dense(2048,kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.00001)))
#     model.add(BatchNormalization())
    # model.add(PReLU(alpha_initializer='zero', weights=None))
#     model.add(Activation('selu'))
#     model.add(Dropout(0.2))
    # model.add(BatchNormalization())
    # model.add(PReLU(alpha_initializer='zero'))
    model.add(Dense(1024,kernel_initializer='uniform' ,input_dim=123)) #,activity_regularizer=regularizers.l1(0.0001)
    model.add(BatchNormalization())
    # model.add(PReLU(alpha_initializer='zero', weights=None))
    model.add(Activation('relu'))
    model.add(Dropout(0.35))
#     model.add(BatchNormalization())
    # model.add(PReLU(alpha_initializer='zero', weights=None))
    
    model.add(Dense(1024,kernel_initializer='uniform')) #,input_dim = 442 #, kernel_regularizer=regularizers.l2(0.0001)
    model.add(BatchNormalization())
    # model.add(PReLU(alpha_initializer='zero', weights=None))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
#     model.add(BatchNormalization())
    # model.add(PReLU(alpha_initializer='zero')) #, weights=None
    # model.add(PReLU(alpha_initializer='zero', weights=None))
    # model.add(Dense(512,activation='relu',kernel_initializer='lecun_uniform'))#,activation='relu'))
    model.add(Dense(256)) #,kernel_initializer='uniform'
    model.add(BatchNormalization())
    # model.add(PReLU(alpha_initializer='zero', weights=None))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
#     model.add(BatchNormalization())
    # model.add(PReLU(alpha_initializer='zero', weights=None))
    # model.add(Dense(256,activation='relu',kernel_initializer='lecun_uniform'))
    model.add(Dense(256))
    model.add(BatchNormalization())
    # model.add(PReLU(alpha_initializer='zero', weights=None))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
#     model.add(BatchNormalization())
    # model.add(BatchNormalization())
    # model.add(PReLU(alpha_initializer='zero', weights=None))
    model.add(Dense(64))
    model.add(BatchNormalization())
    # model.add(PReLU(alpha_initializer='zero', weights=None))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
#     model.add(BatchNormalization())
    # model.add(BatchNormalization())
    # model.add(PReLU(alpha_initializer='zero', weights=None))
    model.add(Dense(64))
    model.add(BatchNormalization())
    # model.add(PReLU(alpha_initializer='zero', weights=None))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
#     model.add(BatchNormalization())
    # model.add(BatchNormalization())
    # model.add(PReLU(alpha_initializer='zero', weights=None))
    model.add(Dense(16,kernel_initializer='uniform'))
    model.add(BatchNormalization())
    # model.add(PReLU(alpha_initializer='zero', weights=None))
    model.add(Activation('relu'))
#     model.add(Dropout(0.1))
#     model.add(BatchNormalization())
    # model.add(BatchNormalization())
    # model.add(PReLU(alpha_initializer='zero', weights=None))
    model.add(Dense(8))
    model.add(BatchNormalization())
    # model.add(PReLU(alpha_initializer='zero', weights=None))
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))
#     model.add(BatchNormalization())

    # model.add(PReLU(alpha_initializer='zero', weights=None))
    model.add(Dense(4))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    # model.add(Dropout(0.2))
    # model.add(PReLU(alpha_initializer='zero', weights=None))
#     model.add(BatchNormalization())

#     # model.add(PReLU(alpha_initializer='zero', weights=None))
#     model.add(Dense(2,kernel_initializer='uniform'))
#     # model.add(BatchNormalization())
#     # model.add(PReLU(alpha_initializer='zero', weights=None))
#     model.add(Activation('relu'))
#     # model.add(Dropout(0.2))
#     model.add(BatchNormalization())

    # model.add(PReLU(alpha_initializer='zero', weights=None))
    # model.add(Dense(128,kernel_initializer='lecun_uniform'))
    # model.add(PReLU(alpha_initializer='zero', weights=None))

 ######################################   
    ##categorical
#     model.add(Dense(2,kernel_initializer='uniform')) #,activation='sigmoid
#     model.add(BatchNormalization())
#     model.add(Activation('softmax'))

    model.add(Dense(2,activation='softmax'))

#     ADAM = Adam(amsgrad=True)
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
 
 ###################################################   
    #regression:
    # model.add(PReLU(alpha_initializer='zero', weights=None))
#     model.add(Dense(2,kernel_initializer='uniform'))
    # model.add(BatchNormalization())
    # model.add(PReLU(alpha_initializer='zero', weights=None))
#     model.add(Activation('relu'))
    # model.add(Dropout(0.2))
#     model.add(BatchNormalization())
    
#     model.add(Dense(1,kernel_initializer='uniform')) #,activation='sigmoid
#     model.add(BatchNormalization())
#     model.add(Activation('sigmoid'))

#     model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
#     ADAM = Adam(amsgrad=False) #amsgrad=False
#     model.compile(loss='binary_crossentropy', optimizer=ADAM, metrics=['accuracy']) #binary_crossentropy
    
#################################################################################################################    
    model.summary()
    batch_size = 16#128 
    early_stopping_monitor = EarlyStopping(patience=20) #,monitor='val_acc',mode='auto' #15
    # keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    callback = GetBest(monitor='val_acc', verbose=1, mode='auto')
    # CW = {0:8.312,1:1.}
    model.fit(X_T, y_T, batch_size=batch_size, epochs=500, validation_data=(x_valid,y_valid), callbacks=[callback,early_stopping_monitor],shuffle=True)#,class_weight=CW)
    
    return model


# In[11]:


##STACKING##
def NN_model2(X_T,y_T,x_valid,y_valid,inputs):    
    model = Sequential()
    model.add(Dense(6144, input_dim=inputs, kernel_initializer='uniform', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
#     model.add(Dropout(0.33))
    model.add(BatchNormalization())
    model.add(Dense(4096, kernel_initializer='uniform', activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
#     model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Dense(512, kernel_initializer='uniform', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
#     model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Dense(64, kernel_initializer='uniform', activation='relu', kernel_regularizer=regularizers.l2(0.00005)))
#     model.add(Dropout(0.2))
    model.add(BatchNormalization())
#     model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
#     model.add(Dropout(0.1))
#     model.add(BatchNormalization())
#     model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
#     model.add(Dropout(0.1))
#     model.add(BatchNormalization())

# model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

 ######################################   
    ##categorical
#     model.add(Dense(2,kernel_initializer='uniform')) #,activation='sigmoid
#     model.add(BatchNormalization())
#     model.add(Activation('softmax'))

#     model.add(Dense(2,activation='softmax'))

#     ADAM = Adam(amsgrad=True)
#     model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
 
 ###################################################   
    #regression:
    # model.add(PReLU(alpha_initializer='zero', weights=None))
    model.add(Dense(8,kernel_initializer='uniform'))
    # model.add(BatchNormalization())
    # model.add(PReLU(alpha_initializer='zero', weights=None))
    model.add(Activation('relu'))
#     model.add(Dropout(0.05))
    model.add(BatchNormalization())
    
#     model.add(Dense(1,kernel_initializer='uniform')) #,activation='sigmoid
#     model.add(BatchNormalization())
#     model.add(Activation('sigmoid'))

    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    opt = Adam(amsgrad=False,decay=1e-8) #amsgrad=False
#     opt = Adamax(lr=0.002,decay=1e-8)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy']) #binary_crossentropy
    
#################################################################################################################    
    model.summary()
    batch_size = 32#128 
    early_stopping_monitor = EarlyStopping(patience=20) #,monitor='val_acc',mode='auto' #15
    # keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    callback = GetBest(monitor='val_acc', verbose=1, mode='auto')
    # CW = {0:8.312,1:1.}
#     model.fit(X_T, y_T, batch_size=batch_size, epochs=500, validation_data=(x_valid,y_valid), callbacks=[callback,early_stopping_monitor],shuffle=True)#,class_weight=CW)
    
    return model


# In[12]:


##STACKING##
def NN_model3(X_T,y_T,x_valid,y_valid,inputs):
    model = Sequential()
    model.add(Dense(8192,activation='relu',input_dim=inputs, kernel_regularizer=regularizers.l2(0.00029))) #411,29
    model.add(Dropout(0.419))
    model.add(BatchNormalization())
    model.add(Dense(8192,activation='relu',kernel_regularizer=regularizers.l2(0.00029))) #411,29
    model.add(Dropout(0.419))
    model.add(BatchNormalization())
    model.add(Dense(128,activation='relu')) #128
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(128,activation='relu')) #128
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(2,activation='softmax'))

 ######################################   
    ##categorical
#     model.add(Dense(2,kernel_initializer='uniform')) #,activation='sigmoid
#     model.add(BatchNormalization())
#     model.add(Activation('softmax'))

#     model.add(Dense(2,activation='softmax'))
    opt = RMSprop(lr=0.00008, decay=1e-7)
#     opt = Nadam(lr=0.0002) #decay1e-8

#     ADAM = Adam(amsgrad=True)
#     model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
 
 ###################################################   

    
#################################################################################################################    
    model.summary()
    batch_size = 32#128 
    early_stopping_monitor = EarlyStopping(patience=20) #,monitor='val_acc',mode='auto' #15
    # keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    callback = GetBest(monitor='val_acc', verbose=1, mode='auto')
    # CW = {0:8.312,1:1.}
#     model.fit(X_T, y_T, batch_size=batch_size, epochs=500, validation_data=(x_valid,y_valid), callbacks=[callback,early_stopping_monitor],shuffle=True)#,class_weight=CW)
    
    return model


# ## Predict Testing data

# In[13]:


def infer(X_test, save_dir, output_dir):
    test_data_size = len(X_test)

    # Load parameters
    print('=====Loading Param from %s=====' % save_dir)
    w = np.loadtxt(os.path.join(save_dir, 'w'))
    b = np.loadtxt(os.path.join(save_dir, 'b'))

    # predict
    z = (np.dot(X_test, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)

    print('=====Write output to %s =====' % output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, 'Sample_Logistic.csv')
    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_):
            f.write('%d,%d\n' %(i+1, v))

    return


# In[14]:


def predict(X_test, save_dir, output_dir):
    test_data_size = len(X_test)

    # Load parameters
    print('=====Loading Param from %s=====' % save_dir)
    model = load_model(os.path.join(save_dir, 'reg_DNN_0.75split_new.h5'))  #model 名稱


    # predict
    ans = model.predict(X_test)
    
    
    #regression  #ensemble
    anss=[]
    for aa in ans:
        if aa[0] < 0.5:
            anss.append(0)
        else:
            anss.append(1)
    print(np.mean(anss))
    
    #categorical:
#     anss = argmax(to_categorical(ans,2),axis=1)


    print('=====Write output to %s =====' % output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, 'reg_DNN_0.75split_new.csv')  ##輸出檔案名稱
    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(anss):
            f.write('%d,%d\n' %(i+1, v))
#             f.write('%d,%d\n' %(i+1, v[1]))  #categorical


    return model


# In[15]:


def predict2(X_test, save_dir, output_dir):
    test_data_size = len(X_test)

    # Load parameters
    print('=====Loading Param from %s=====' % save_dir)
    model = load_model(os.path.join(save_dir, 'clf_DNN_0.75split_newo.h5'))  #model 名稱


    # predict
    ans = model.predict(X_test)
    
    
    anss=[]
    #categorical:
    for v in ans:
        if v[1]<0.5:
            anss.append(0)
        else:
            anss.append(1)
#     anss = argmax(to_categorical(ans,2),axis=1)


    print('=====Write output to %s =====' % output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, 'clf_DNN_086_aug_newo.csv')  ##輸出檔案名稱
    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(anss):
            f.write('%d,%d\n' %(i+1, v))
#             f.write('%d,%d\n' %(i+1, v[1]))  #categorical


    return model


# ## Params

# * unbalaced data weight
# * data augmentation
# * softmax retry with big model

# In[21]:


def main():
    # Load feature and label
#     X_all, Y_all, X_test = load_data('data/data_old/X_train', 'data/data_old/Y_train', 'data/data_old/X_test')
    
    #softmax categorical clf
#     with open('model/clf_colo.pickle','rb') as f:
#         cols = pickle.load(f)
#     len(clf_col)
    
    #regressor
    with open('model/reg_colo.picke', 'rb') as f:
        cols = pickle.load(f)



    X_all, Y_all, X_test = load_data('data/train_X', 'data/train_Y', 'data/test_X',cols)
    # Normalization
    X_all, X_test = normalize(X_all, X_test)
    valid_set_percentage = 0.75 #train percentage
    X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)
    dfT = pd.DataFrame(X_train,columns=None)
    dfA = pd.DataFrame(Y_train,columns=None)
    dfM = dfT.merge(dfA,how='inner',left_index=True,right_index=True)
    dfMM = dfM[dfM['0_y'] == 1 ]
    times = (dfM.shape[0]-dfMM.shape[0])/dfMM.shape[0]
    # dfMM.shape[0]  #知道有幾個row 如果小於3.1 就兩倍，否則三倍
    if times < 3.2:
        times = 0
    else:
        times = 1
    for i in range(times):
        dfM = dfM.append(dfMM)
    dfT = dfM.drop(columns='0_y')
    dfA = dfM.filter(['0_y'])

    X_train = np.array(pd.concat([dfT],axis=1))
    Y_train = np.array(pd.concat([dfA],axis=1))
    
    
    #softmax categorical need to add these two
    Y_train = to_categorical(Y_train, num_classes=2)
    Y_valid = to_categorical(Y_valid, num_classes=2)
    
#     X_train, Y_train = _shuffle(X_train, Y_train)
    print(X_train.shape,Y_train.shape)
#     print(Y_train)
    
    # To train or to infer
    trains = False
    infers = True
    
    if trains:
#         train(X_all, Y_all, 'model/model_old/')
#         train(X_all, Y_all, 'model/')
        model = NN_model3(X_train,Y_train,X_valid,Y_valid,len(cols))
        model.save('model/clf_DNN_0.75split_newo.h5')  #儲存檔案名稱
    elif infers:
#         infer(X_test,  'model/model_old/', 'result/result_old/')
#         infer(X_test,  'model/', 'result/')
        model = predict2(X_test, 'model/', 'result/')
    else:
        print("Error: Argument --train or --infer not found")
    return model

if __name__ == '__main__':
#     opts = 'kk'
#     parser = argparse.ArgumentParser(description='Logistic Regression with Gradient Descent Method')
#     group = parser.add_mutually_exclusive_group()
#     group.add_argument('--train', action='store_true', default=True,
#                         dest='train', help='Input --train to Train')
#     group.add_argument('--infer', action='store_true',default=False,
#                         dest='infer', help='Input --infer to Infer')
#     parser.add_argument('--train_data_path', type=str,
#                         default='data/data_old/X_train', dest='train_data_path',
#                         help='Path to training data')
#     parser.add_argument('--train_label_path', type=str,
#                         default='data/data_old/X_train', dest='train_label_path',
#                         help='Path to training data\'s label')
#     parser.add_argument('--test_data_path', type=str,
#                         default='data/X_test', dest='test_data_path',
#                         help='Path to testing data')
#     parser.add_argument('--save_dir', type=str,
#                         default='model/', dest='save_dir',
#                         help='Path to save the model parameters')
#     parser.add_argument('--output_dir', type=str,
#                         default='model/', dest='output_dir',
#                         help='Path to save the model parameters')
#     opts = parser.parse_args()
    model = main()
    #Using epoch 00027 with val_acc: 0.86292=>cat_newo  clf_DNN_0.75split_newo.h5
    #Using epoch 00017 with val_acc: 0.86046=>reg new  reg_DNN_0.75split_new.h5


# In[20]:


model.save_weights('model/clf_NN_weights.h5')
model = model.load_weights('model/clf_NN_weights.h5')


# In[113]:


X_all, Y_all, X_test = load_data('data/train_X', 'data/train_Y', 'data/test_X')
# Normalization
X_all, X_test = normalize(X_all, X_test)
valid_set_percentage = 0.75 #train percentage
X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)
dfT = pd.DataFrame(X_train,columns=None)
dfA = pd.DataFrame(Y_train,columns=None)
dfM = dfT.merge(dfA,how='inner',left_index=True,right_index=True)
dfM


# In[117]:


dfMM = dfM[dfM['0_y'] == 1 ]
times = (dfM.shape[0]-dfMM.shape[0])/dfMM.shape[0]
# dfMM.shape[0]  #知道有幾個row 如果小於3.1 就兩倍，否則三倍
if times < 3.1:
    times = 2
else:
    times = 3


# In[110]:


for i in range(times):
    dfM = dfM.append(dfMM)
dfT = dfM.drop(columns='0_y')
dfA = dfM.filter(['0_y'])

X_train = np.array(pd.concat([dfT],axis=1))
Y_train = np.array(pd.concat([dfA],axis=1))


# In[224]:


model = load_model('model_old/regression_DNN_0.75split2.h5')
model.summary()


# In[221]:


X_all, Y_all, X_test = load_data('data/train_X', 'data/train_Y', 'data/test_X')
X_all, X_test = normalize(X_all, X_test)
ans = model.predict(X_all)
ans


# In[222]:


# anss = argmax(to_categorical(ans,2),axis=1)
y=[]
for v in ans:
    if v<0.5:
        y.append(0)
    else:
        y.append(1)
#     y.append(v[1])

print(np.mean(y))
ans = []
for i,v in enumerate(y):
    ans.append(abs(v-Y_all[i][0]))
#     print(i)
acc = ans.count(0)/len(ans)
print("ACC:",acc)
#ACC: 0.8516812528788577
#ACC: 0.8530893010686648
#ACC: 0.868063020177513
#0.17358189244802064
#ACC: 0.8599244494947944


# In[213]:


model.save('model/final_clf.h5')


# In[104]:


# for i in ans:
#     print(round(1-i[0],0))
kk=[]
for i in ans:
    if i[0] < 0.5:
        kk.append(0)
    else:
        kk.append(1)
#     kk.append(i[0])
#     print (i[0])
np.mean(kk)


# In[54]:


kk=[]
from numpy import argmax
anss = argmax(to_categorical(ans,2),axis=1)
for i in anss:
    kk.append(i[1])
#     print(i[1])
np.mean(kk)
# ans[0][0]


# In[86]:


Y_all.shape

