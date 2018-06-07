
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


def cmapRefine(cmap):
    cmap['likes'] = cmap['liked'] = cmap['lk'] = 'like'
    cmap['wierd'] = 'weird'
    cmap['mee'] = 'me'
    cmap['hooo'] = 'hoo'
    cmap['sooon'] = cmap['soooon'] = 'soon'
    cmap['goodd'] = cmap['gud'] = 'good'
    cmap['bedd'] = 'bed'
    cmap['badd'] = 'bad'
    cmap['sadd'] = 'sad'
    cmap['kool'] = 'cool'
    cmap['yess'] = 'yes'
    cmap['teh'] = cmap['da'] = cmap['tha'] = 'the'
    cmap['evar'] = 'ever'
    cmap['pleasee'] = 'please'
    cmap['soo'] = 'so'
    cmap['noo'] = 'no'
    cmap['ilove'] = 'i love'
    cmap['liek'] = cmap['lyk'] = cmap['lik'] = cmap['lke'] = cmap['likee'] = 'like'
    cmap['madd'] = 'mad'
    cmap['lovelovelove'] = 'love love love'
    cmap['redd'] = 'red'
    cmap['2moro'] = cmap['2mrow'] = cmap['2morow'] = cmap['2morrow'] = cmap['2morro'] = cmap['2mrw'] = cmap['2moz'] = 'tomorrow'
    cmap['babee'] = 'babe'
    cmap['tiredd'] = 'tired'
    cmap['w00t'] = 'woot'
    cmap['srsly'] = 'seriously'
    cmap['4ever'] = cmap['4eva'] = 'forever'
    cmap['neva'] = 'never'
    cmap['2day'] = 'today'
    cmap['theree'] = 'there'
    cmap['thee'] = 'the'
    cmap['homee'] = 'home'
    cmap['hatee'] = 'hate'
    cmap['boredd'] = 'bored'
    cmap['lovee'] = cmap['loove'] = cmap['looove'] = cmap['loooove'] = cmap['looooove'] = cmap['loooooove'] = cmap['loves'] = cmap['loved'] = cmap['wuv'] = cmap['loovee'] = cmap['lurve'] = cmap['lov'] = cmap['luvs'] = 'love'
    cmap['lovelove'] = 'love love'
    cmap['godd'] = 'god'
    cmap['xdd'] = 'xd'
    cmap['itt'] = 'it'
    cmap['lul'] = cmap['lool'] = 'lol'
    cmap['sista'] = 'sister'
    cmap['heree'] = 'here'
    cmap['cutee'] = 'cute'
    cmap['lemme'] = 'let me'
    cmap['mrng'] = 'morning'
    cmap['gd'] = 'good'
    cmap['thx'] = cmap['thnx'] = cmap['thanx'] = cmap['thankx'] = cmap['thnk'] = 'thanks'
    cmap['nite'] = 'night'
    cmap['dnt'] = 'dont'
    cmap['rly'] = 'really'
    cmap['gt'] = 'get'
    cmap['lat'] = 'late'
    cmap['dam'] = 'damn'
    cmap['cuz'] = cmap['bcuz'] = cmap['becuz'] = 'because'
    cmap['iz'] = 'is'
    cmap['aint'] = 'am not'
    cmap['fav'] = 'favorite'
    cmap['eff'] = cmap['fk'] = cmap['fuk'] = cmap['fuc'] = 'fuck'
    cmap['ppl'] = 'people'
    cmap['boi'] = 'boy'
    cmap['4ward'] = 'forward'
    cmap['4give'] = 'forgive'
    cmap['b4'] = 'before'
    cmap['jaja'] = cmap['jajaja'] = cmap['jajajaja'] = 'haha'
    cmap['woho'] = cmap['wohoo'] = 'woo hoo'
    cmap['2gether'] = 'together'
    cmap['2nite'] = cmap['2night'] = 'tonight'
    cmap['tho'] = 'though'
    cmap['kno'] = 'know'
    cmap['grl'] = 'girl'
    cmap['xoxox'] = cmap['xox'] = cmap['xoxoxo'] = cmap['xoxoxox'] = cmap['xoxoxoxo'] = cmap['xoxoxoxoxo'] = 'xoxo'
    cmap['wrk'] = 'work'
    cmap['loveyou'] = cmap['loveya'] = cmap['loveu'] = 'love you'
    cmap['jst'] = 'just'
    cmap['2go'] = 'to go'
    cmap['2b'] = 'to be'
    cmap['gr8'] = cmap['gr8t'] = cmap['gr88'] = 'great'
    cmap['str8'] = 'straight'
    cmap['twiter'] = 'twitter'
    cmap['iloveyou'] = 'i love you'
    cmap['xboxe3'] = 'eTHREE'
    cmap['jammin'] = 'jamming'
    cmap['onee'] = 'one'
    cmap['1st'] = 'first'
    cmap['2nd'] = 'second'
    cmap['3rd'] = 'third'
    cmap['inet'] = 'internet'
    cmap['geting'] = 'getting'
    cmap['4get'] = 'forget'
    cmap['4got'] = 'forgot'
    cmap['4real'] = 'for real'
    cmap['mah'] = 'my'
    cmap['r8'] = 'rate'
    cmap['l8'] = 'late'
    cmap['w8'] = 'wait'
    cmap['m8'] = 'mate'
    cmap['h8'] = 'hate'
    cmap['any1'] = 'anyone'
    cmap['every1'] = cmap['evry1'] = 'everyone'
    cmap['some1'] = cmap['sum1'] = 'someone'
    cmap['no1'] = 'no one'
    cmap['ah1n1'] = cmap['h1n1'] = 'hONEnONE'
    cmap['yr'] = cmap['yrs'] = cmap['years'] = 'year'
    cmap['hr'] = cmap['hrs'] = cmap['hours'] = 'hour'
    cmap['go2'] = cmap['goto'] = 'go to'
    cmap['4u'] = 'for you'
    cmap['4me'] = 'for me'
    cmap['2u'] = 'to you'
    cmap['cnt'] = 'cant'
    cmap['fone'] = cmap['phonee'] = 'phone'
    cmap['f1'] = 'fONE'
    cmap['yu'] = 'you'
    cmap['l8ter'] = cmap['l8tr'] = cmap['l8r'] = 'later'
    cmap['min'] = cmap['mins'] = cmap['minutes'] = 'minute'
    cmap['recomend'] = 'recommend'
    for key, value in cmap.items():
        if not key.isalpha():
            if key[-1:] == 'k':
                cmap[key] = '_n'
            if key[-2:]=='st' or key[-2:]=='nd' or key[-2:]=='rd' or key[-2:]=='th':
                cmap[key] = '_ord'
            if key[-2:]=='am' or key[-2:]=='pm' or key[-3:]=='min' or key[-4:]=='mins'  or key[-2:]=='hr' or key[-3:]=='hrs' or key[-1:]=='h' or key[-4:]=='hour' or key[-5:]=='hours' or key[-2:]=='yr' or key[-3:]=='yrs'or key[-3:]=='day' or key[-4:]=='days'or key[-3:]=='wks':
                cmap[key] = '_time'


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

