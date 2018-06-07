
# coding: utf-8

# In[1]:


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
import glove
import gensim
from gensim.models.word2vec import *
from keras.preprocessing.text import *
from keras.preprocessing.sequence import *
from keras.utils import *
import pickle


# In[2]:


def get_Test(path,size,type_):
    lines = Testing_Preprocessing(path,type_)
    
#     w2v_path = 'models/w2v_'+str(size)+'.pkl'
    dicts_path = 'models/dicts_'+str(size)+'.pkl'
    
    dicts = read_dicts(dicts_path)
    Change_fromMap(lines, dicts)
    
    lines = Line_padding(lines, '_', max_length)
#     w2v = pickle.load(open('models/w2v_'+str(type_)+'.pkl','rb'))
#     gensim.models.keyedvectors.
    w2v = gensim.models.KeyedVectors.load_word2vec_format('models/w2v_'+str(type_)+'.pkl',binary=True)
#     w2v = Word2Vec.load(w2v_path)
    Covert_w2v(lines, w2v)
    return lines


# In[3]:


def Testing_Preprocessing(path,type_):
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


# In[4]:


def read_dicts(path):
    with open(path, 'rb') as f:
        dicts = pickle.load(f)
    return dicts


# In[5]:


def Change_fromMap(lines, dicts, iter=2):
    Map_dicts(dicts)
    for it in range(iter):
        for i, s in enumerate(lines):
            s0 = []
            for j, w in enumerate(s):
                if w in dicts and w[0] != '_':
                    s0 = s0 + dicts[w].split()
                elif w[0] == '_':
                    s0 = s0 + [w]
            lines[i] = [w for w in s0 if w]


# In[6]:


def Map_dicts(dicts):
    dicts['likes'] = dicts['liked'] = dicts['lk'] = 'like'
    dicts['wierd'] = 'weird'
    dicts['mee'] = 'me'
    dicts['hooo'] = 'hoo'
    dicts['sooon'] = dicts['soooon'] = 'soon'
    dicts['goodd'] = dicts['gud'] = 'good'
    dicts['bedd'] = 'bed'
    dicts['badd'] = 'bad'
    dicts['sadd'] = 'sad'
    dicts['kool'] = 'cool'
    dicts['yess'] = 'yes'
    dicts['teh'] = dicts['da'] = dicts['tha'] = 'the'
    dicts['evar'] = 'ever'
    dicts['pleasee'] = 'please'
    dicts['soo'] = 'so'
    dicts['noo'] = 'no'
    dicts['ilove'] = 'i love'
    dicts['liek'] = dicts['lyk'] = dicts['lik'] = dicts['lke'] = dicts['likee'] = 'like'
    dicts['madd'] = 'mad'
    dicts['lovelovelove'] = 'love love love'
    dicts['redd'] = 'red'
    dicts['2moro'] = dicts['2mrow'] = dicts['2morow'] = dicts['2morrow'] = dicts['2morro'] = dicts['2mrw'] = dicts['2moz'] = 'tomorrow'
    dicts['babee'] = 'babe'
    dicts['tiredd'] = 'tired'
    dicts['w00t'] = 'woot'
    dicts['srsly'] = 'seriously'
    dicts['4ever'] = dicts['4eva'] = 'forever'
    dicts['neva'] = 'never'
    dicts['2day'] = 'today'
    dicts['theree'] = 'there'
    dicts['thee'] = 'the'
    dicts['homee'] = 'home'
    dicts['hatee'] = 'hate'
    dicts['boredd'] = 'bored'
    dicts['dnt'] = 'dont'
    dicts['rly'] = 'really'
    dicts['gt'] = 'get'
    dicts['lat'] = 'late'
    dicts['lovee'] = dicts['loove'] = dicts['looove'] = dicts['loooove'] = dicts['looooove'] = dicts['loooooove'] = dicts['loves'] = dicts['loved'] = dicts['wuv'] = dicts['loovee'] = dicts['lurve'] = dicts['lov'] = dicts['luvs'] = 'love'
    dicts['lovelove'] = 'love love'
    dicts['godd'] = 'god'
    dicts['xdd'] = 'xd'
    dicts['itt'] = 'it'
    dicts['lul'] = dicts['lool'] = 'lol'
    dicts['sista'] = 'sister'
    dicts['heree'] = 'here'
    dicts['cutee'] = 'cute'
    dicts['lemme'] = 'let me'
    dicts['mrng'] = 'morning'
    dicts['gd'] = 'good'
    dicts['thx'] = dicts['thnx'] = dicts['thanx'] = dicts['thankx'] = dicts['thnk'] = 'thanks'
    dicts['nite'] = 'night'
    dicts['dam'] = 'damn'
    dicts['cuz'] = dicts['bcuz'] = dicts['becuz'] = 'because'
    dicts['iz'] = 'is'
    dicts['aint'] = 'am not'
    dicts['fav'] = 'favorite'
    dicts['eff'] = dicts['fk'] = dicts['fuk'] = dicts['fuc'] = 'fuck'
    dicts['ppl'] = 'people'
    dicts['boi'] = 'boy'
    dicts['4ward'] = 'forward'
    dicts['4give'] = 'forgive'
    dicts['b4'] = 'before'
    dicts['jaja'] = dicts['jajaja'] = dicts['jajajaja'] = 'haha'
    dicts['woho'] = dicts['wohoo'] = 'woo hoo'
    dicts['2gether'] = 'together'
    dicts['2nite'] = dicts['2night'] = 'tonight'
    dicts['tho'] = 'though'
    dicts['kno'] = 'know'
    dicts['grl'] = 'girl'
    dicts['xoxox'] = dicts['xox'] = dicts['xoxoxo'] = dicts['xoxoxox'] = dicts['xoxoxoxo'] = dicts['xoxoxoxoxo'] = 'xoxo'
    dicts['wrk'] = 'work'
    dicts['loveyou'] = dicts['loveya'] = dicts['loveu'] = 'love you'
    dicts['jst'] = 'just'
    dicts['2go'] = 'to go'
    dicts['xboxe3'] = 'eTHREE'
    dicts['jammin'] = 'jamming'
    dicts['onee'] = 'one'
    dicts['1st'] = 'first'
    dicts['2nd'] = 'second'
    dicts['3rd'] = 'third'
    dicts['2b'] = 'to be'
    dicts['gr8'] = dicts['gr8t'] = dicts['gr88'] = 'great'
    dicts['str8'] = 'straight'
    dicts['twiter'] = 'twitter'
    dicts['iloveyou'] = 'i love you'
    dicts['inet'] = 'internet'
    dicts['geting'] = 'getting'
    dicts['4get'] = 'forget'
    dicts['4got'] = 'forgot'
    dicts['4real'] = 'for real'
    dicts['mah'] = 'my'
    dicts['r8'] = 'rate'
    dicts['l8'] = 'late'
    dicts['w8'] = 'wait'
    dicts['m8'] = 'mate'
    dicts['h8'] = 'hate'
    dicts['any1'] = 'anyone'
    dicts['every1'] = dicts['evry1'] = 'everyone'
    dicts['some1'] = dicts['sum1'] = 'someone'
    dicts['no1'] = 'no one'
    dicts['ah1n1'] = dicts['h1n1'] = 'hONEnONE'
    dicts['yr'] = dicts['yrs'] = dicts['years'] = 'year'
    dicts['hr'] = dicts['hrs'] = dicts['hours'] = 'hour'
    dicts['go2'] = dicts['goto'] = 'go to'
    dicts['4u'] = 'for you'
    dicts['4me'] = 'for me'
    dicts['2u'] = 'to you'
    dicts['cnt'] = 'cant'
    dicts['fone'] = dicts['phonee'] = 'phone'
    dicts['f1'] = 'fONE'
    dicts['yu'] = 'you'
    dicts['l8ter'] = dicts['l8tr'] = dicts['l8r'] = 'later'
    dicts['min'] = dicts['mins'] = dicts['minutes'] = 'minute'
    dicts['recomend'] = 'recommend'
    for key, value in dicts.items():
        if not key.isalpha():
            if key[-1:] == 'k':
                dicts[key] = '_n'
            if key[-2:]=='st' or key[-2:]=='nd' or key[-2:]=='rd' or key[-2:]=='th':
                dicts[key] = '_ord'
            if key[-2:]=='am' or key[-2:]=='pm' or key[-3:]=='min' or key[-4:]=='mins'  or key[-2:]=='hr' or key[-3:]=='hrs' or key[-1:]=='h' or key[-4:]=='hour' or key[-5:]=='hours' or key[-2:]=='yr' or key[-3:]=='yrs'or key[-3:]=='day' or key[-4:]=='days'or key[-3:]=='wks':
                dicts[key] = '_time'


# In[7]:


def Line_padding(lines, value, lengthMax):
    line_max = 0
    for i, s in enumerate(lines):
        line_max = max(len(s), line_max)
    line_max = max(line_max, lengthMax)
    for i, s in enumerate(lines):
        lines[i] = (['_r'] * max(0, line_max - len(s)) + s)[-lengthMax:]
    return lines


# In[8]:


def Covert_w2v(lines, w2v):
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w in w2v.wv:
                lines[i][j] = w2v.wv[w]
            else:
                lines[i][j] = w2v.wv['_r']


# In[9]:


def to_label(x):
    if x<=0:
        return 0
    if x>0:
        return 1


# In[10]:


def to_label2(x):
    if x>=0.5:
        return 1
    if x<0.5:
        return 0


# In[11]:


test_path = sys.argv[1]#'data/testing_data.txt'

semi_r = True #True , False
semi_c = True #True , False
ens = 'plus' #plus/times
predict_path = sys.argv[2]#'result/reproduce4.csv'#'result/re_'+str(semi_r)+'2_'+str(semi_c)+'2_'+ens+'.csv'
# print(test_X.shape)


# In[12]:


# predict_path


# ## Model

# In[13]:


max_length=39 # 39,40
size = 128 #128,256
type_ ='reg' #reg,clf

test_X = get_Test(test_path,size,type_)
# print(len(test_X))
test_X = np.array(test_X)


# In[14]:


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
    model.load_weights('models/LSTM_LSTM_reg_128_semi2.h5')
res_reg =  model.predict(test_X, batch_size=512)


# In[15]:


max_length=40 # 39,40
size = 256 #128,256
type_ ='clf' #reg,clf
test_X = get_Test(test_path,size,type_)
# print(len(test_X))
test_X = np.array(test_X)


# In[16]:


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
    model.load_weights('models/BiLSTM_BiLSTM_BiLSTM_clf_256_semi2.h5')
res_clf =  model.predict(test_X, batch_size=512)


# ## Ens

# In[17]:


acc_reg = 0.8088
acc_reg_semi = 0.8176#1=>0.8094 2=>8176
acc_clf = 0.8193
acc_clf_semi = 0.8189#1=>0.8166 2=>8189


# In[18]:



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


# In[22]:


# df


# In[19]:


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


# In[20]:


if ens == 'plus':
    df['label'] = df['label'].map(to_label)
else:
    df['label'] = df['label'].map(to_label2)
    


# In[21]:


df = df.filter(['id','label'])
df = df.astype('int')
df.to_csv(predict_path,index=False)
print(df)
print(predict_path)

