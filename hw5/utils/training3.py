
# coding: utf-8

# In[64]:


import os, sys
import numpy as np
from random import shuffle
from math import log, floor
import pandas as pd
import tensorflow as tf
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
#import glove
#import gensim
from gensim.models.word2vec import Word2Vec
import gensim
from keras.preprocessing.text import *
from keras.preprocessing.sequence import *
from keras.utils import *
import pickle


# In[65]:


def preprocessTestingData(path,size,type_):
    lines = readTestData(path,type_)

    cmap_path = 'model/cmap_'+str(size)+'.pkl'
    w2v_path = 'model/word2vec_'+str(size)+'.pkl'
    cmap = loadPreprocessCmap(cmap_path)
    transformByConversionMap(lines, cmap)
    
    lines = padLines(lines, '_', max_length)
    #w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path,encoding='ascii')
    w2v = Word2Vec.load(w2v_path)
    w2v.wv.save_word2vec_format('models/w2v_'+str(type_)+'.pkl',binary=True)
#     pickle.dump(w2v,open('models/w2v_'+str(type_)+'.pkl','wb'))
    transformByWord2Vec(lines, w2v)
    return lines


# In[66]:


def readTestData(path,type_):
    _lines = []
    if type_ == 'reg':
        stemmer = gensim.parsing.porter.PorterStemmer()
    with open(path, 'r', encoding='utf_8') as f:
        for i, line in enumerate(f):
            if i:
                start = int(np.log10(max(1, i-1))) + 2
                if type_ == 'reg':
                    line = stemmer.stem_sentence(line)
                _lines.append(line[start:].split())
    return _lines


# In[67]:


def loadPreprocessCmap(path):
    with open(path, 'rb') as f:
        cmap = pickle.load(f)
    return cmap


# In[68]:


def transformByConversionMap(lines, cmap, iter=2):
    cmapRefine(cmap)
    for it in range(iter):
        for i, s in enumerate(lines):
            s0 = []
            for j, w in enumerate(s):
                if w in cmap and w[0] != '_':
                    s0 = s0 + cmap[w].split()
                elif w[0] == '_':
                    s0 = s0 + [w]
            lines[i] = [w for w in s0 if w]


# In[69]:




# In[70]:


def padLines(lines, value, maxlen):
    maxlinelen = 0
    for i, s in enumerate(lines):
        maxlinelen = max(len(s), maxlinelen)
    maxlinelen = max(maxlinelen, maxlen)
    for i, s in enumerate(lines):
        lines[i] = (['_r'] * max(0, maxlinelen - len(s)) + s)[-maxlen:]
    return lines


# In[71]:


def transformByWord2Vec(lines, w2v):
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w in w2v.wv:
                lines[i][j] = w2v.wv[w]
            else:
                lines[i][j] = w2v.wv['_r']


# In[72]:


def to_label(x):
    if x<=0:
        return 0
    if x>0:
        return 1


# In[73]:


def to_label2(x):
    if x>=0.5:
        return 1
    if x<0.5:
        return 0


# In[74]:


test_path = sys.argv[1]#'data/testing_data.txt'

semi_r = True #True , False
semi_c = True #True , False
ens = 'plus' #plus/times
predict_path = sys.argv[2]#'result/'+str(semi_r)+'2_'+str(semi_c)+'2_'+ens+'.csv'
# print(test_X.shape)


# In[75]:


# predict_path


# ## Model

# In[76]:


max_length=39 # 39,40
size = 128 #128,256
type_ ='reg' #reg,clf

test_X = preprocessTestingData(test_path,size,type_)
# print(len(test_X))
test_X = np.array(test_X)


# In[77]:


#reg
model = Sequential()
model.add(LSTM(128,input_shape=(39,128),activation='tanh',dropout=0.55,return_sequences=True,kernel_initializer='uniform',recurrent_dropout=0.55))
model.add(LSTM(128,activation='tanh',dropout=0.55,return_sequences=False,kernel_initializer='uniform',recurrent_dropout=0.55))
model.add(BatchNormalization())
model.add(Dense(64,activation='relu',kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.1)))
model.add(Dropout(0.55))
model.add(Dense(1,activation='sigmoid'))
model.summary()
if semi_r == False:
    model.load_weights('model/LSTM_LSTM_reg_128.h5')
else:
    model.load_weights('model/LSTM_LSTM_reg_128_semi2.h5')
res_reg =  model.predict(test_X, batch_size=512)


# In[78]:


max_length=40 # 39,40
size = 256 #128,256
type_ ='clf' #reg,clf
test_X = preprocessTestingData(test_path,size,type_)
# print(len(test_X))
test_X = np.array(test_X)


# In[79]:


#clf
model = Sequential()
model.add(Bidirectional(LSTM(128,dropout=0.7,recurrent_dropout=0.7,
                            return_sequences=True,kernel_initializer='lecun_normal'), input_shape=(40,256)))
model.add(Bidirectional(LSTM(128,dropout=0.7,recurrent_dropout=0.7,return_sequences=True,kernel_initializer='lecun_normal')))
model.add(Bidirectional(LSTM(128,dropout=0.7,recurrent_dropout=0.7,return_sequences=False,kernel_initializer='lecun_normal')))
model.add(BatchNormalization())
model.add(Dense(64,activation='selu',kernel_initializer='lecun_normal'))
model.add(Dropout(0.7))
model.add(BatchNormalization())
model.add(Dense(64,activation='selu',kernel_initializer='lecun_normal'))
model.add(Dropout(0.7))
model.add(Dense(2,activation='softmax'))
model.summary()
if semi_c == False:
    model.load_weights('model/BiLSTM_BiLSTM_BiLSTM_clf_256.h5')
else:
    model.load_weights('model/BiLSTM_BiLSTM_BiLSTM_clf_256_semi2.h5')
res_clf =  model.predict(test_X, batch_size=512)


# ## Ens

# In[80]:


acc_reg = 0.8088
acc_reg_semi = 0.8176#1=>0.8094 2=>8176
acc_clf = 0.8193
acc_clf_semi = 0.8189#1=>0.8166 2=>8189


# In[81]:



df = pd.merge(pd.DataFrame(res_reg),pd.DataFrame(res_clf),how='inner',right_index=True,left_index=True)
df.columns = ['reg','clf0','clf1']
# df = df.drop(['clf0'],axis=1)
if ens == 'plus':
    print('+')
    df['reg'] = df['reg']*2-1
    df['clf1'] = df['clf1']*2-1
else:
    pass
df['id']=df.index


# In[82]:


# df['label'] = (df[1]+df['0_y'])/2

if (semi_r == True) and (semi_c == True):
    if ens == 'times':
        df['label'] = (df['reg']*acc_reg_semi*df['clf1']*acc_clf_semi)**(1/(acc_reg_semi+acc_clf_semi))
    else:
        df['label'] = (df['reg']*acc_reg_semi+df['clf1']*acc_clf_semi)/(acc_reg_semi+acc_clf_semi)
        print("TT+")
elif (semi_r == True) and (semi_c == False):
    if ens == 'times':
        df['label'] = (df['reg']*acc_reg_semi*df['clf1']*acc_clf)**(1/(acc_reg_semi+acc_clf))
    else:
        df['label'] = (df['reg']*acc_reg_semi+df['clf1']*acc_clf)/(acc_reg_semi+acc_clf)
        print('TF+')
elif (semi_r == False) and (semi_c == True):
    if ens == 'times':
        df['label'] = (df['reg']*acc_reg*df['clf1']*acc_clf_semi)**(1/(acc_reg+acc_clf_semi))
    else:
        df['label'] = (df['reg']*acc_reg+df['clf1']*acc_clf_semi)/(acc_reg+acc_clf_semi)
        print("FT+")
else:
    if ens == 'times':
        df['label'] = (df['reg']*acc_reg*df['clf1']*acc_clf)**(1/(acc_reg+acc_clf))
        print("FF*")
    else:
        df['label'] = (df['reg']*acc_reg+df['clf1']*acc_clf)/(acc_reg+acc_clf)
        print("FF+")


# In[83]:


if ens == 'plus':
    df['label'] = df['label'].map(to_label)
else:
    df['label'] = df['label'].map(to_label2)
    


# In[84]:


df = df.filter(['id','label'])
df = df.astype('int')
df.to_csv(predict_path,index=False)
print(predict_path)
df

