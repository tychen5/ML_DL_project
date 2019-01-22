
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
from keras import *
from keras.engine.topology import *
from keras.optimizers import *
import keras
import glove
import gensim
from gensim.models.word2vec import *
from keras.preprocessing.text import *
from keras.preprocessing.sequence import *
from keras.utils import *
from sklearn.model_selection import *
import random
from random import shuffle
import re
from collections import Counter
from keras.utils.generic_utils import *
from keras import regularizers
import string
import unicodedata as udata
import pickle
from keras.applications import *
from keras.preprocessing.image import *
K.set_image_dim_ordering('tf')


from utils import feature, utils


# In[2]:


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


# In[3]:


def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
#     print(X.shape, Y.shape)
    return (X[randomize], Y[randomize])


# ## load prediction data

# In[12]:


type_ = 'clf' #reg/clf
ans_reg = pickle.load(open('result/semi_ans_reg.pkl','rb'))
ans_clf = pickle.load(open('result/semi_ans_clf.pkl','rb'))


# In[5]:


# valid_Y = pickle.load(open('data/'+type_+'_validY.pkl','rb'))
# valid_Y2 = pickle.load(open('data/clf_validY.pkl','rb'))
# print(valid_Y[0],ans_reg[0],ans_clf[0])


# In[13]:


ans_df = pd.DataFrame(ans_reg)
ans = pd.DataFrame(ans_clf)
ans_df = pd.merge(ans_df,ans,how='inner',left_index=True,right_index=True)
ans_df.columns = ['reg','clf0','clf1']
ans_df['fin'] = (ans_df["reg"]*0.8088+ans_df['clf1']*0.8193)/(0.8088+0.8193)
ans_df


# In[14]:


def to_label(x):
    if x<0.1:
        return 0
    if x>0.9:
        return 1


# In[15]:


range_ = ans_df.index[(ans_df['fin'] > 0.9) | (ans_df['fin'] < 0.1)].tolist()
ans_df['label'] = ans_df['fin'].map(to_label)
semi_Y = np.array(ans_df.iloc[range_,4])
semi_Y.shape , semi_Y


# In[16]:



semi_X = np.load('data/semiX_'+type_+'.npy')
semi_X = semi_X[range_]
train_X = pickle.load(open('data/'+type_+'_trainX.pkl','rb'))
train_Y = pickle.load(open('data/'+type_+'_trainY.pkl','rb'))
valid_X = pickle.load(open('data/'+type_+'_validX.pkl','rb'))
valid_Y = pickle.load(open('data/'+type_+'_validY.pkl','rb'))
if type_ == 'clf':
    semi_Y = to_categorical(semi_Y, num_classes=2)
print(semi_X.shape , train_X.shape , valid_X.shape)


# In[17]:


train_X = np.concatenate((semi_X,train_X))
train_X = np.concatenate((valid_X,train_X))
train_Y = np.concatenate((semi_Y,train_Y))
train_Y = np.concatenate((valid_Y,train_Y))
train_X, train_Y = _shuffle(train_X,train_Y)
print(train_Y.shape, valid_Y.shape)


# ## Re-Train

# In[11]:


model = load_model('model/LSTM_LSTM_reg_128.h5_all.h5')
opt=Adam(lr=1e-4,decay=1e-6,amsgrad=False)
# for layer in model.layers[:2]:  #freeze first two layers
#     layer.trainable = False
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

batchSize=1024
patien=30
epoch=300
saveP = 'model/LSTM_LSTM_reg_128_semi2.h5'
logD = './logs/'
history = History()

callback=[
    EarlyStopping(patience=patien,monitor='val_acc',verbose=1),
    ModelCheckpoint(saveP,monitor='val_loss',verbose=1,save_best_only=True, save_weights_only=True),
    TensorBoard(log_dir=logD+'events.epochs'+str(epoch)),
    history,
]
model.fit(train_X, train_Y,
                epochs=epoch,
                batch_size=batchSize,
                shuffle=True,
                validation_data=(valid_X,valid_Y),
                callbacks=callback, 
                class_weight='auto'
                )
model.save(saveP+"_all.h5")
#loss: 0.2642 - acc: 0.8997 - val_loss: 0.5088 - val_acc: 0.8094
# loss: 0.2767 - acc: 0.8822 - val_loss: 0.4147 - val_acc: 0.8176


# In[18]:


#classifier
opt=Adamax(lr=2e-4, decay=1e-6)
model = load_model('model/BiLSTM_BiLSTM_BiLSTM_clf_256.h5_all.h5')
for layer in model.layers[:1]:  #freeze first two layers
    layer.trainable = False
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

batchSize=1024
patien=35
epoch=250
saveP = 'model/BiLSTM_BiLSTM_BiLSTM_clf_256_semi2.h5'
logD = './logs/'
history = History()

callback=[
    EarlyStopping(patience=patien,monitor='val_loss',verbose=1),
    ModelCheckpoint(saveP,monitor='val_acc',verbose=1,save_best_only=True, save_weights_only=True),
    TensorBoard(log_dir=logD+'events.epochs'+str(epoch)),
    history,
]
model.fit(train_X, train_Y,
                epochs=epoch,
                batch_size=batchSize,
                shuffle=True,
                validation_data=(valid_X,valid_Y),
                callbacks=callback, 
                class_weight='auto'
                )
model.save(saveP+"_all.h5")
# loss: 0.4197 - acc: 0.8095 - val_loss: 0.4125 - val_acc: 0.8193
#0.8159, 0.8166
# loss: 0.2456 - acc: 0.9093 - val_loss: 0.4829 - val_acc: 0.8189

