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
    def on_train_begin(self, logs={}):
        self.tr_losses = []
        self.val_losses = []
        self.tr_accs = []
        self.val_accs = []

    def on_epoch_end(self, epoch, logs={}):
        self.tr_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.tr_accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))


# In[50]:


Line_ori = []

def Start_mapping(lines):
    dicts = {}
    #     stemmer = gensim.parsing.porter.PorterStemmer()
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            #             w = stemmer.stem_sentence(w) #takeoff pruls
            dicts[w] = w
    print('Mapping Length:', len(dicts))
    return dicts

def Find_dict(lines):
    freq = {}
    for s in lines:
        for w in s:
            if w in freq:
                freq[w] += 1
            else:
                freq[w] = 1
    return freq


def Change_du1(lines, dicts):
    # freq = Find_dict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '':
                continue
            w = re.sub(r'(([a-z])\2{2,})$', r'\g<2>\g<2>', w)
            s[j] = re.sub(r'(([a-cg-kmnp-ru-z])\2+)$', r'\g<2>', w)
            dicts[Line_ori[i][j]] = s[j]
        lines[i] = s







def Change_simbols(lines, dicts):
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '': continue
            Q_counts, A_counts, Dots_count = w.count('?'), w.count('!'), w.count('.')
            if Q_counts:
                s[j] = '_?'
            elif Dots_count >= 2:
                s[j] = '_...'
            elif A_counts >= 5:
                s[j] = '_!!!'
            elif A_counts >= 1:
                s[j] = '_!'
            elif A_counts >= 3:
                s[j] = '_!!'
            elif Dots_count >= 1:
                s[j] = '_.'
            dicts[Line_ori[i][j]] = s[j]
        lines[i] = s


def Change_sound(lines, dicts):
    for i, s in enumerate(lines):
        s = [(''.join(c for c in udata.normalize('NFD', w) if udata.category(c) != 'Mn')) for w in s]
        for j, w in enumerate(s):
            dicts[Line_ori[i][j]] = s[j]
        lines[i] = s
    clist = '0123456789abcdefghijklmnopqrstuvwxyz.!?'
    for i, s in enumerate(lines):
        s = [''.join([c for c in w if c in clist]) for w in s]
        for j, w in enumerate(s):
            dicts[Line_ori[i][j]] = s[j]
        lines[i] = s

def Change_rared(lines, dicts, min_count=16):  # 替換稀有字
    freq = Find_dict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '':
                continue
            if freq[w] < min_count:
                s[j] = '_r'
            dicts[Line_ori[i][j]] = s[j]
        lines[i] = s
def Chage_single(lines, dicts, minfreq=512):
    freq = Find_dict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '':
                continue
            if freq[w] > minfreq:
                continue
            w1 = re.sub(r"s$", r'', w)
            w2 = re.sub(r"es$", r'', w)
            w3 = re.sub(r"ies$", r'y', w)
            f0, f1, f2, f3 = freq.get(w, 0), freq.get(w1, 0), freq.get(w2, 0), freq.get(w3, 0)
            fm = max(f0, f1, f2, f3)
            if fm == f0:
                pass
            elif fm == f1:
                s[j] = w1  # ;
            elif fm == f2:
                s[j] = w2  # ;
            else:
                s[j] = w3  # ;
            dicts[Line_ori[i][j]] = s[j]
    lines[i] = s


def Chane_formal(lines, dicts):  # 非正常用語(口頭用語)
    freq = Find_dict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '':
                continue
            if w == 'u':
                lines[i][j] = 'you'
            if w == 'luv':
                lines[i][j] = 'love'
            if w == 'dis':
                lines[i][j] = 'this'
            if w == 'dat':
                lines[i][j] = 'that'
            w1 = re.sub(r"in$", r'ing', w)
            w2 = re.sub(r"n$", r'ing', w)
            f0, f1, f2 = freq.get(w, 0), freq.get(w1, 0), freq.get(w2, 0)
            fm = max(f0, f1, f2)
            if fm == f0:
                pass
            elif fm == f1:
                s[j] = w1  # ;
            else:
                s[j] = w2  # ;
            dicts[Line_ori[i][j]] = s[j]
        lines[i] = s

def Change_weird(lines, dicts):
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '':
                continue
            if w[0] == '_':
                continue
            if w == '2':
                s[j] = 'to'
            elif w.isnumeric():
                s[j] = '_n'
            dicts[Line_ori[i][j]] = s[j]
        lines[i] = s


def Change_du2(lines, dicts):
    # freq = Find_dict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '':
                continue
            s[j] = re.sub(r'^(([a-km-z])\2+)', r'\g<2>', w)
            dicts[Line_ori[i][j]] = s[j]
        lines[i] = s


def Change_du3(lines, dicts, minfreq=64):
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '':
                continue
            w = re.sub(r'(([a-z])\2{2,})', r'\g<2>\g<2>', w)
            s[j] = re.sub(r'(([ahjkquvwxyz])\2+)', r'\g<2>', w)
            dicts[Line_ori[i][j]] = s[j]
        lines[i] = s
    freq = Find_dict(lines)
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w == '':
                continue
            if freq[w] > minfreq:
                continue
            if w == 'too':
                continue
            w1 = re.sub(r'(([a-z])\2+)', r'\g<2>', w)
            f0, f1 = freq.get(w, 0), freq.get(w1, 0)
            fm = max(f0, f1)
            if fm == f0:
                pass
            else:
                s[j] = w1  # ;
            dicts[Line_ori[i][j]] = s[j]
        lines[i] = s






# def dummyCode(lines, dicts): #常見字，可以不用這個function
#     for i, s in enumerate(lines):
#         lines[i] = s

def pads(lines, lengthMax=38):  # 要pad到多長
    for i, s in enumerate(lines):
        lines[i] = [w for w in s if w]
    for i, s in enumerate(lines):
        lines[i] = s[:lengthMax]


def Preprocessing_firstTime(lines):  # 第一次前處理
    global Line_ori
    Line_ori = lines[:]
    dicts = Start_mapping(Line_ori)
    Change_sound(lines, dicts)
    Change_simbols(lines, dicts)
    Change_weird(lines, dicts)
    Change_du1(lines, dicts)
    Change_du2(lines, dicts)
    Change_du3(lines, dicts)
    Chane_formal(lines, dicts)
    Chage_single(lines, dicts)
    Change_rared(lines, dicts)
    #     dummyCode(lines, dicts)
    pads(lines)
    return lines, dicts


def Get_data(path, label=True, train=True):
    _lines, _labels = [], []
    stemmer = gensim.parsing.porter.PorterStemmer()  # clf才要註解，reg不要註解掉
    with open(path, 'r', encoding='utf_8') as f:
        for line in f:
            line = stemmer.stem_sentence(line)  # clf才要註解，reg不要註解掉
            if not train:
                label = False
                line = line.split(',')[1:]
                line = ''.join(map(str, line))
            if label:
                _labels.append(int(line[0]))
                line = line[10:-1]
            else:
                line = line[:-1]
            _lines.append(line.split())
    if label:
        return _lines, _labels
    else:
        return _lines


def Line_padding(lines, value, lengthMax):  # 對各行進行padding
    line_max = 0
    for i, s in enumerate(lines):
        line_max = max(len(s), line_max)
    line_max = max(line_max, lengthMax)
    for i, s in enumerate(lines):
        lines[i] = (['_r'] * max(0, line_max - len(s)) + s)[-lengthMax:]
    return lines


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




def Testing_Preprocessing(path):
    _lines = []
    stemmer = gensim.parsing.porter.PorterStemmer()  # clf才要註解，reg不要註解掉
    with open(path, 'r', encoding='utf_8') as f:
        for i, line in enumerate(f):
            if i:
                start = int(np.log10(max(1, i - 1))) + 2
                line = stemmer.stem_sentence(line)  # clf才要註解，reg不要註解掉
                _lines.append(line[start:].split())
    return _lines

def Covert_w2v(lines, w2v):  # 利用w2v轉換
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w in w2v.wv:
                lines[i][j] = w2v.wv[w]
            else:
                lines[i][j] = w2v.wv['_r']
# def saving(y, path, id_start=0):  #儲存成CSV，dummyCode
#     pd.DataFrame([[i+id_start, int(y[i])] for i in range(y.shape[0])],
#                  columns=['id', 'label']).to_csv(path, index=False)

def saving_words(lines, path):  # 儲存corpus
    with open(path, 'w', encoding='utf_8') as f:
        for line in lines:
            f.write(' '.join(line) + '\n')




def saving_dicts(dicts, path):  # 前處理的dicts儲存
    with open(path, 'wb') as f:
        pickle.dump(dicts, f)


def read_dicts(path):  # 讀取前處理dicts
    with open(path, 'rb') as f:
        dicts = pickle.load(f)
    return dicts


def read_words(path):  # 讀取corpus ，dummyCode
    lines = []
    with open(path, 'r', encoding='utf_8') as f:
        for line in f:
            lines.append(line.split())
    return lines


def eliminate_symbols(lines):  # 移除雜七雜八符號
    rs = {'_!!!', '_.', '_!', '_!!', '_...', '_?'}
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w in rs:
                s[j] = ''
        lines[i] = [w for w in x if w]


def eliminate_dup_ln(lines):
    lineset = set({})
    for line in lines:
        lineset.add(' '.join(line))
    for i, line in enumerate(lineset):
        lines[i] = line.split()
    del lines[-(len(lines) - len(lineset)):]
    return lineset


def shuffleData(lines, labels):
    for i, s in enumerate(lines):
        lines[i] = (s, labels[i])
    np.random.shuffle(lines)
    for i, s in enumerate(lines):
        labels[i] = s[1]
        lines[i] = s[0]


def Map_dicts(dicts):  # dont change to test fn
    dicts['wierd'] = 'weird'
    dicts['1ce'] = 'once'
    dicts['26y4u'] = 'too sexy for you'
    dicts['teh'] = dicts['da'] = dicts['tha'] = 'the'
    dicts['2day'] = 'today'
    dicts['likes'] = dicts['liked'] = dicts['lk'] = 'like'
    dicts['pleasee'] = 'please'
    dicts['2mor'] = 'tomorrow'
    dicts['2nite'] = 'tonight'
    dicts['ilove'] = 'i love'
    dicts['afaik'] = 'as far as i know'
    dicts['bil'] = 'boss is listening'
    dicts['cu'] = 'see you'
    dicts['tiredd'] = 'tired'
    dicts['liek'] = dicts['lyk'] = dicts['lik'] = dicts['lke'] = dicts['likee'] = 'like'
    dicts['cuz'] = 'because'
    dicts['ez'] = 'easy'
    dicts['sooon'] = dicts['soooon'] = 'soon'
    dicts['boredd'] = 'bored'
    dicts['godd'] = 'god'
    dicts['xdd'] = 'xd'
    dicts['grt'] = 'great'
    dicts['lul'] = dicts['lool'] = 'lol'
    dicts['iluvu'] = 'i love you'
    dicts['ic'] = 'i see'
    dicts['jic'] = 'just in case'
    dicts['4ever'] = dicts['4eva'] = 'forever'
    dicts['ldn'] = 'london'
    dicts['msg'] = 'message'
    dicts['woho'] = dicts['wohoo'] = 'woo hoo'
    dicts['2gether'] = 'together'
    dicts['2nite'] = dicts['2night'] = 'tonight'
    dicts['2day'] = 'today'
    dicts['mtg'] = 'meeting'
    dicts['mth'] = 'month'
    dicts['nvm'] = 'never mind'
    dicts['goodd'] = dicts['gud'] = 'good'
    dicts['plz'] = 'please'
    dicts['ru'] = 'are you'
    dicts['thx'] = 'thanks'
    dicts['lemme'] = 'let me'
    dicts['mrng'] = 'morning'
    dicts['yr'] = 'your'
    dicts['aka'] = 'also known as'
    dicts['nite'] = 'night'
    dicts['dnt'] = 'dont'
    dicts['4give'] = 'forgive'
    dicts['b4'] = 'before'
    dicts['tho'] = 'though'
    dicts['kno'] = 'know'
    dicts['grl'] = 'girl'
    dicts['boi'] = 'boy'
    dicts['rly'] = 'really'
    dicts['gt'] = 'get'
    dicts['lat'] = 'late'
    dicts['dam'] = 'damn'
    dicts['4ward'] = 'forward'
    dicts['wrk'] = 'work'
    dicts['f2f'] = 'face to face'
    dicts['geting'] = 'getting'
    dicts['str8'] = 'straight'
    dicts['iyq'] = 'i like you'
    dicts['iloveyou'] = 'i love you'
    dicts['aint'] = 'am not'
    dicts['fav'] = 'favorite'
    dicts['ppl'] = 'people'
    dicts['xboxe3'] = 'eTHREE'
    dicts['ne1'] = 'anyone'
    dicts['4get'] = 'forget'
    dicts['4got'] = 'forgot'
    dicts['4real'] = 'for real'
    dicts['2go'] = 'to go'
    dicts['2b'] = 'to be'
    dicts['gr8'] = dicts['gr8t'] = dicts['gr88'] = 'great'
    dicts['np'] = 'no problem'
    dicts['mah'] = 'my'
    dicts['r8'] = 'rate'
    dicts['l8ter'] = dicts['l8tr'] = dicts['l8r'] = 'later'
    dicts['cnt'] = 'cant'
    dicts['fone'] = dicts['phonee'] = 'phone'
    dicts['f1'] = 'fONE'
    dicts['recomend'] = 'recommend'
    dicts['any1'] = 'anyone'
    dicts['every1'] = dicts['evry1'] = 'everyone'
    dicts['1st'] = 'first'
    dicts['2nd'] = 'second'
    dicts['3rd'] = 'third'
    dicts['inet'] = 'internet'
    dicts['yu'] = 'you'
    dicts['l8'] = 'late'
    dicts['wel'] = 'well'
    dicts['sum1'] = 'someone'
    dicts['h8'] = 'hate'
    dicts['yr'] = dicts['yrs'] = dicts['years'] = 'year'
    dicts['hr'] = dicts['hrs'] = dicts['hours'] = 'hour'
    dicts['min'] = dicts['mins'] = dicts['minutes'] = 'minute'
    dicts['go2'] = dicts['goto'] = 'go to'
    dicts['some1'] = dicts['sum1'] = 'someone'
    dicts['no1'] = 'no one'
    dicts['4u'] = 'for you'
    dicts['4me'] = 'for me'
    dicts['2u'] = 'to you'


# In[4]:


def preprocess(label_path, nolabel_path, test_path, size=256):
    print('Preprocessing...')
    labeled_lines, labels = Get_data(label_path)
    Line_ori = []
    nolabel_lines = Get_data(nolabel_path, label=False)
    Line_ori = []
    test = Get_data(test_path, train=False)
    Line_ori = []
    lines = labeled_lines + nolabel_lines + test
    Line_ori = []
    lines, dicts = Preprocessing_firstTime(lines)
    Line_ori = []
    corpus_path = 'model/corpus_128.txt'  #
    Line_ori = []
    dicts_path = 'model/dicts_128.pkl'  #
    Line_ori = []
    w2v_path = 'model/word2vec_128.pkl'  #
    Line_ori = []
    saving_words(lines, corpus_path)
    Line_ori = []
    saving_dicts(dicts, dicts_path)
    Line_ori = []
    Change_fromMap(lines, dicts)
    Line_ori = []
    eliminate_dup_ln(lines)
    Line_ori = []

    #     lines = [sent for sent in stemmer.stem_documents(lines)]
    print('Training word2vec...')
    model = Word2Vec(lines, size=size, min_count=5, iter=24,
                     workers=24)  # change=>size=512,min_count=5,window=5,workers=12
    model.save(w2v_path)


# In[53]:


def get_train(label_path, nolabel_path, punctuation=True):  # 前處理訓練資料

    lines, labels = Get_data(label_path)
    corpus_path = 'model/corpus_128.txt'
    dicts_path = 'model/dicts_128.pkl'
    w2v_path = 'model/word2vec_128.pkl'
    lines = Get_data(corpus_path, label=False)[:len(lines)]
    shuffleData(lines, labels)
    labels = np.array(labels)

    dicts = read_dicts(dicts_path)
    Change_fromMap(lines, dicts)
    if not punctuation:
        eliminate_symbols(lines)

    lines = Line_padding(lines, '_', max_length)  # 最大長度39
    w2v = Word2Vec.load(w2v_path)
    Covert_w2v(lines, w2v)
    return lines, labels


# In[52]:


def get_Test(path):  # 前處理測試資料(無label者)
    lines = Testing_Preprocessing(path)

    dicts_path = 'model/dicts_128.pkl'  # 改
    w2v_path = 'model/word2vec_128.pkl'  # 改
    dicts = read_dicts(dicts_path)
    Change_fromMap(lines, dicts)

    lines = Line_padding(lines, '_', max_length)
    w2v = Word2Vec.load(w2v_path)
    Covert_w2v(lines, w2v)
    return lines


# In[7]:


def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid


# In[8]:


def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    #     print(X.shape, Y.shape)
    return X[randomize], Y[randomize]


# In[9]:


# def shuffleX(X):
#     randomize = np.arange(len(X))
#     np.random.shuffle(randomize)
# #     print(X.shape, Y.shape)
#     return X[randomize]


# In[54]:


train_path = 'data/training_label.txt'  # label
test_path = 'data/testing_data.txt'
train_no_label = 'data/training_nolabel.txt'
max_length = 39  # 39,40
size = 128  # 128,256
# with open(train_path) as f:
#     train = f.readlines()
# train_Y = [sen.strip().split(" +++$+++ ")[0]for sen in train]
# train_X = [sen.strip().split(" +++$+++ ")[1]for sen in train]
######################################################################
preprocess(train_path, train_no_label, test_path, size)  # first time only
######################################################################
# stemmer = gensim.parsing.porter.PorterStemmer()
test_X = get_Test(test_path)
semi_X = get_Test(train_no_label)
train_X, train_Y = get_train(train_path, test_path)
train_X = np.array(train_X)
train_Y = np.array(train_Y)
# label
semi_X += test_X
shuffle(semi_X)
print(len(test_X), len(semi_X))

# In[11]:


# len(semi_X)


# In[11]:


#########################################semi2###############
# train_Y = to_categorical(train_Y, num_classes=2)
# valid_X = train_X
# valid_Y = train_Y
############################only retrain semi data$$$$$$$$$$$$$$$


# In[11]:


# softmax categorical need to add this one#########
train_Y = to_categorical(train_Y, num_classes=2)
#########################
train_X, train_Y, valid_X, valid_Y = split_valid_set(train_X, train_Y, 0.9)
print(train_X.shape, train_Y.shape, valid_X.shape)

# In[22]:


pickle.dump(train_X, open('data/clf_trainX.pkl', 'wb'), protocol=-1)
pickle.dump(train_Y, open('data/clf_trainY.pkl', 'wb'), protocol=-1)
pickle.dump(valid_X, open('data/clf_validX.pkl', 'wb'), protocol=-1)
pickle.dump(valid_Y, open('data/clf_validY.pkl', 'wb'), protocol=-1)

# In[23]:


# train_X = pickle.load(open('data/reg_trainX.pkl','rb'))
# train_Y = pickle.load(open('data/reg_trainY.pkl','rb'))
# valid_X = pickle.load(open('data/reg_validX.pkl','rb'))
# valid_Y = pickle.load(open('data/reg_validY.pkl','rb'))


# In[55]:


# pickle.dump
# df_semi_x_clf = pd.DataFrame(semi_X)
# df_semi_x_clf
semi_x_clf = np.array(semi_X)
print(semi_x_clf.shape)
# pickle.dump(semi_x_clf,open('data/semiX_clf.pkl','wb'),protocol=-1)
np.save('data/semiX_reg.npy', semi_x_clf)

# In[62]:


semi_x_clf = np.load('data/semiX_clf.npy')
semi_x_clf.shape

# In[56]:


# semi_x_clf.shape

# In[49]:


# semi_x_clf.shape

# In[64]:


# model = load_model('model/BiLSTM_BiLSTM_BiLSTM_clf_256.h5_all.h5')
# ans_clf = model.predict(semi_x_clf,batch_size=1024)


# In[65]:


# pickle.dump(ans_clf,open('result/semi_ans_clf.pkl','wb'),protocol=-1)


# ## Model

# In[28]:


# reg
model = Sequential()
model.add(LSTM(128, input_shape=(39, 128), activation='tanh', dropout=0.55, return_sequences=True,
               kernel_initializer='uniform', recurrent_dropout=0.55))
model.add(LSTM(128, activation='tanh', dropout=0.55, return_sequences=False, kernel_initializer='uniform',
               recurrent_dropout=0.55))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu', kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.1)))
model.add(Dropout(0.55))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# In[19]:


# clf
model = Sequential()
model.add(Bidirectional(LSTM(128, dropout=0.7, recurrent_dropout=0.7,
                             return_sequences=True, kernel_initializer='lecun_normal'), input_shape=(max_length, size)))
model.add(Bidirectional(
    LSTM(128, dropout=0.7, recurrent_dropout=0.7, return_sequences=True, kernel_initializer='lecun_normal')))
model.add(Bidirectional(
    LSTM(128, dropout=0.7, recurrent_dropout=0.7, return_sequences=False, kernel_initializer='lecun_normal')))
model.add(BatchNormalization())
model.add(Dense(64, activation='selu', kernel_initializer='lecun_normal'))
model.add(Dropout(0.7))
model.add(BatchNormalization())
model.add(Dense(64, activation='selu', kernel_initializer='lecun_normal'))
model.add(Dropout(0.7))
model.add(Dense(2, activation='softmax'))
model.summary()

# In[29]:


# regression
opt = Adam(decay=1e-20, amsgrad=True)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

batchSize = 256
patien = 25
epoch = 250
saveP = 'model/LSTM_LSTM_reg_128.h5'
logD = './logs/'
history = History()

callback = [
    EarlyStopping(patience=patien, monitor='val_acc', verbose=1),
    ModelCheckpoint(saveP, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True),
    TensorBoard(log_dir=logD + 'events.epochs' + str(epoch)),
    history,
]
model.fit(train_X, train_Y,
          epochs=epoch,
          batch_size=batchSize,
          shuffle=True,
          validation_data=(valid_X, valid_Y),
          callbacks=callback,
          class_weight='auto'
          )
model.save(saveP + "_all.h5")
#  loss: 0.4475 - acc: 0.7971 - val_loss: 0.4216 - val_acc: 0.8088


# In[20]:


# classifier
opt = Adamax(decay=1e-20)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

batchSize = 256
patien = 10
epoch = 200
saveP = 'model/BiLSTM_BiLSTM_BiLSTM_clf_256.h5'
logD = './logs/'
history = History()

callback = [
    EarlyStopping(patience=patien, monitor='val_loss', verbose=1),
    ModelCheckpoint(saveP, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True),
    TensorBoard(log_dir=logD + 'events.epochs' + str(epoch)),
    history,
]
model.fit(train_X, train_Y,
          epochs=epoch,
          batch_size=batchSize,
          shuffle=True,
          validation_data=(valid_X, valid_Y),
          callbacks=callback,
          class_weight='auto'
          )
model.save(saveP + "_all.h5")
# loss: 0.4197 - acc: 0.8095 - val_loss: 0.4125 - val_acc: 0.8193


# ***

# In[15]:


# classification
# model = load_model('model/biLSTM_biLSTM_clf_1024_2.h5_all.h5')
#
#
# batchSize=1024 #64
# patien=3
# epoch=50
# saveP = 'model/biLSTM_biLSTM_clf_1024_2_semi.h5'
# logD = './logs/'
# history = History()
#
# callback=[
#     EarlyStopping(patience=patien,monitor='val_acc',verbose=1),
#     ModelCheckpoint(saveP,monitor='val_loss',verbose=1,save_best_only=True, save_weights_only=True),
#     TensorBoard(log_dir=logD+'events.epochs'+str(epoch)),
#     history,
# ]
# shuffle(semi_X)
# for i in range(30):
#     X_semi = np.array(semi_X[i*int(len(semi_X)/30):(i+1)*int(len(semi_X)/30)])
#     res_semi = model.predict(X_semi,batch_size=1024)
#     a1X=[]
#     a1Y=[]
#     a2X=[]
#     a2Y=[]
#     for ii,ans in enumerate(res_semi):
#         if ans[0]>0.95:
# #             train_X = np.concatenate((train_X,X_semi[ii]))
#             a1X.append(ii)
#             a1Y.append([1,0])
# #             train_Y = np.concatenate((train_Y,np.array([1,0])))
#         elif ans[0]<0.05:
#             a2X.append(ii)
#             a2Y.append([0,1])
# #             train_X = np.concatenate((train_X,X_semi[ii]))
# #             train_Y = np.concatenate((train_Y,np.array([0,1])))
#     try:
#         train_X = np.concatenate((X_semi[a1X],train_X))
#         train_Y = np.concatenate((np.array(a1Y),train_Y))
#     except Exception as e:
#         print('append a1 error',str(e))
#         pass
#     try:
#         train_X, train_Y = _shuffle(train_X,train_Y) #ram會到62+80GB #可能要try 以免memory error
#     except Exception as e:
#         print("shuffle a1 error",str(e))
#         pass
#     try:
#         train_X = np.concatenate((train_X,X_semi[a2X]))
#         train_Y = np.concatenate((train_Y,np.array(a2Y)))
#     except Exception as e:
#         print("append a2 error",str(e))
#         pass
#     try:
#         train_X, train_Y = _shuffle(train_X,train_Y)
#     except Exception as e:
#         print("shuffle a2 error",str(e))
#         pass
#     opt = [Adam(),Adam(amsgrad=True),Adamax(),Nadam(),RMSprop()] #decay1e-8
#     opt = random.choice(opt)
#     model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#     model.fit(train_X, train_Y,
#                     epochs=epoch,
#                     batch_size=batchSize,
#                     shuffle=True,
#                     validation_data=(valid_X,valid_Y),
#                     callbacks=callback,
#                     class_weight='auto'
#                     )
#     model.save(saveP+"_all.h5")
#
#
# # In[12]:
#
#
# #classification##semi2
# model = load_model('model/biLSTM_biLSTM_clf_1024_2.h5_all.h5')
#
#
# batchSize=1024 #64
# patien=3
# epoch=50
# saveP = 'model/biLSTM_biLSTM_clf_1024_2_semi2.h5'
# logD = './logs/'
# history = History()
#
# callback=[
#     EarlyStopping(patience=patien,monitor='val_acc',verbose=1),
#     ModelCheckpoint(saveP,monitor='val_loss',verbose=1,save_best_only=True, save_weights_only=True),
#     TensorBoard(log_dir=logD+'events.epochs'+str(epoch)),
#     history,
# ]
# shuffle(semi_X)
# for i in range(5):
#     X_semi = np.array(semi_X[i*int(len(semi_X)/5):(i+1)*int(len(semi_X)/5)])
#     res_semi = model.predict(X_semi,batch_size=1024)
#     a1X=[]
#     a1Y=[]
#     a2X=[]
#     a2Y=[]
#     for ii,ans in enumerate(res_semi):
#         if ans[0]>0.95:
# #             train_X = np.concatenate((train_X,X_semi[ii]))
#             a1X.append(ii)
#             a1Y.append([1,0])
# #             train_Y = np.concatenate((train_Y,np.array([1,0])))
#         elif ans[0]<0.05:
#             a2X.append(ii)
#             a2Y.append([0,1])
# #             train_X = np.concatenate((train_X,X_semi[ii]))
# #             train_Y = np.concatenate((train_Y,np.array([0,1])))
# #     try:
# #         train_X = X_semi[a1X]#np.concatenate((X_semi[a1X],train_X))
# #         train_Y = np.array(a1Y)#np.concatenate((np.array(a1Y),train_Y))
# #         train_X, train_Y = _shuffle(train_X,train_Y) #ram會到62+80GB #可能要try 以免memory error
#
# #     except Exception, e:
# #         print('init a1 error',str(e))
# #         pass
# #     try:
# #     except Exception, e:
# #         print("shuffle a1 error",str(e))
# #         pass
#     try:
#         train_X = np.concatenate(( X_semi[a1X],X_semi[a2X]))
#         train_Y = np.concatenate((np.array(a1Y),np.array(a2Y)))
#     except Exception as e:
#         print("append a2 error",str(e))
#         pass
#     try:
#         train_X, train_Y = _shuffle(train_X,train_Y)
#     except Exception as e:
#         print("shuffle a2 error",str(e))
#         pass
#     opt = [Adam(lr=1e-4, decay=1e-6),Adam(lr=1e-4, decay=1e-6,amsgrad=True),
#            Adamax(lr=2e-4, decay=1e-6),Nadam(lr=2e-4, schedule_decay=0.04),RMSprop(lr=1e-4, decay=1e-6)] #decay1e-8
#     opt = random.choice(opt)
#     model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#     model.fit(train_X, train_Y,
#                     epochs=epoch,
#                     batch_size=batchSize,
#                     shuffle=True,
#                     validation_data=(valid_X,valid_Y),
#                     callbacks=callback,
#                     class_weight='auto'
#                     )
#     model.save(saveP+"_all.h5")
#
#
# # In[88]:
#
#
# # X_semi.shape, train_X.shape
# # semi_X = shuffleX(semi_X)
# # semi_X = shuffle(semi_X)
# # kk = [1,2,3]
# # kk2 = [4,5,6]
# # kk += kk2
# # random.shuffle(kk)
# # print(kk)
# # np.concatenate((train_X,np.expand_dims(X_semi[ii],axis=0))).shape
# # kk = np.array([1,2,3])
# # kk.expand_dims
# # kk = np.expand_dims(kk,axis=0)
# # kk.shape
# i=[]
# kk=[0,1]
# i.append(kk)
# i.append(kk)
# i.append([0,1])
# np.array(i).shape
#
#
# # In[85]:
#
#
# X_semi[[0,1]].shape
#
#
# # In[28]:
#
#
#
# res_semi = model.predict(valid_X,batch_size=1024)
#
#
# # In[30]:
#
#
# countA=0
# countB=0
# kk=[]
# for ans in res_semi:
#     if ans[0] > 0.95:
#         countA+=1
# #         kk.append(X_test[i])
#     elif ans[0] <0.05:
#         countB+=1
# #         kk.append(X_test[i])
# print(countA+countB)
#
#
# # In[38]:
#
#
# np.array([0,1])
#
#
# # In[36]:
#
#
# train_Y.shape
#
#
# # In[40]:
#
#
# countA=0
# countB=0
# kk=[]
# for i in range(len(semi)):
#     if semi[i][0] > 0.95:
#         countA+=1
#         kk.append(X_test[i])
#     elif semi[i][0] <0.05:
#         countB+=1
#         kk.append(X_test[i])
# print(countA+countB, np.array(kk).shape)
#
#
# # In[38]:
#
#
# X_test[0]
#
#
# # ## Model Archeitecture
#
# # In[10]:
#
#
# #classification sequence_input = Input(shape=(max_length,size)) rnn = Bidirectional(LSTM(512,activation='tanh',
# dropout=0.5,recurrent_dropout=0.5,return_sequences=True,kernel_initializer='lecun_normal'))(sequence_input) # bn =
# BatchNormalization()(rnn) rnn = Bidirectional(LSTM(512,activation='tanh',dropout=0.5,recurrent_dropout=0.5,
# return_sequences=False,kernel_initializer='lecun_normal'))(rnn) bn = BatchNormalization()(rnn)
#
# dense = Dense(512,kernel_initializer='lecun_normal')(bn) #,kernel_regularizer=regularizers.l2(0.02)
# bn = BatchNormalization()(dense)
# activation = Activation('selu')(bn)
# dropout = Dropout(0.65)(activation)
#
# dense = Dense(512,kernel_initializer='lecun_normal')(dropout) # ,kernel_regularizer=regularizers.l2(0.02)
# bn = BatchNormalization()(dense)
# activation = Activation('selu')(bn)
# dropout = Dropout(0.65)(activation)
#
# dense = Dense(128,kernel_initializer='lecun_normal')(dropout)
# bn = BatchNormalization()(dense)
# activation = Activation('selu')(bn)
# dropout = Dropout(0.2)(activation)
#
#
# dense = Dense(128,kernel_initializer='lecun_normal')(dropout)
# bn = BatchNormalization()(dense)
# activation = Activation('selu')(bn)
# dropout = Dropout(0.2)(activation)
#
# output = Dense(2,activation='softmax')(dropout) #clf
# model = Model(sequence_input,output)
# model.summary()
#
#
# # In[10]:
#
#
# #classification sequence_input = Input(shape=(max_length,size)) rnn = Bidirectional(LSTM(512,activation='tanh',
# dropout=0.5,recurrent_dropout=0.5,return_sequences=True,kernel_initializer='lecun_normal'))(sequence_input) # bn =
# BatchNormalization()(rnn) rnn = Bidirectional(LSTM(512,activation='tanh',dropout=0.5,recurrent_dropout=0.5,
# return_sequences=False,kernel_initializer='lecun_normal'))(rnn) bn = BatchNormalization()(rnn)
#
# dense = Dense(512,kernel_initializer='lecun_normal')(bn) #,kernel_regularizer=regularizers.l2(0.02)
# bn = BatchNormalization()(dense)
# activation = Activation('selu')(bn)
# dropout = Dropout(0.65)(activation)
#
# dense = Dense(512,kernel_initializer='lecun_normal')(dropout) # ,kernel_regularizer=regularizers.l2(0.02)
# bn = BatchNormalization()(dense)
# activation = Activation('selu')(bn)
# dropout = Dropout(0.65)(activation)
#
# dense = Dense(128,kernel_initializer='lecun_normal')(dropout)
# bn = BatchNormalization()(dense)
# activation = Activation('selu')(bn)
# dropout = Dropout(0.2)(activation)
#
#
# dense = Dense(128,kernel_initializer='lecun_normal')(dropout)
# bn = BatchNormalization()(dense)
# activation = Activation('selu')(bn)
# dropout = Dropout(0.2)(activation)
#
# output = Dense(2,activation='softmax')(dropout) #clf
# model = Model(sequence_input,output)
# model.summary()
# #loss: 0.3587 - acc: 0.8422 - val_loss: 0.3661 - val_acc: 0.8385=>biLSTM_biLSTM_clf_1024_2.h5
#
#
# # In[20]:
#
#
# #classification sequence_input = Input(shape=(max_length,size)) rnn = Bidirectional(LSTM(256,activation='tanh',
# dropout=0.55,recurrent_dropout=0.5,return_sequences=True,kernel_initializer='he_uniform'))(sequence_input) rnn =
# Bidirectional(GRU(256,activation='tanh',dropout=0.6,recurrent_dropout=0.6,return_sequences=False,
# kernel_initializer='he_uniform'))(rnn) bn = BatchNormalization()(rnn)
#
# dense = Dense(512,kernel_initializer='lecun_normal',kernel_regularizer=regularizers.l2(0.01))(bn)
# dropout = Dropout(0.65)(dense)
# bn = BatchNormalization()(dropout)
# activation = Activation('selu')(bn)
#
# dense = Dense(512,kernel_initializer='lecun_normal',kernel_regularizer=regularizers.l2(0.01))(bn) #
# dropout = Dropout(0.65)(dense)
# bn = BatchNormalization()(dropout)
# activation = Activation('selu')(bn)
#
# dense = Dense(128,kernel_initializer='lecun_normal')(bn)
# bn = BatchNormalization()(dense)
# activation = Activation('selu')(bn)
# dropout = Dropout(0.2)(activation)
#
#
# dense = Dense(128,kernel_initializer='lecun_normal')(bn)
# bn = BatchNormalization()(dense)
# activation = Activation('selu')(bn)
# dropout = Dropout(0.2)(activation)
#
# output = Dense(2,activation='softmax')(dropout) #clf
# model = Model(sequence_input,output)
# model.summary()
# ##loss: 0.3229 - acc: 0.8630 - val_loss: 0.3898 - val_acc: 0.8381=>biLSTM_biLSTM_clf_1024
#
#
# # In[10]:
#
#
# #regression sequence_input = Input(shape=(max_length,size)) rnn = Bidirectional(LSTM(1024,activation='tanh',
# dropout=0.6,recurrent_dropout=0.55,return_sequences=True,kernel_initializer='he_uniform'))(sequence_input) rnn =
# Bidirectional(GRU(512,activation='tanh',dropout=0.7,recurrent_dropout=0.65,return_sequences=False,
# kernel_initializer='he_uniform'))(rnn) bn = BatchNormalization()(rnn)
#
# dense = Dense(512,activation='selu',kernel_initializer='lecun_normal',kernel_regularizer=regularizers.l2(0.0009))(
# bn) dropout = Dropout(0.75)(dense) bn = BatchNormalization()(dropout) dense = Dense(256,activation='softplus',
# kernel_regularizer=regularizers.l2(0.0009))(bn) #,kernel_initializer='lecun_normal' dropout = Dropout(0.8)(dense)
# bn = BatchNormalization()(dropout)
#
# # dense = Dense(512,activation='softplus',kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(
# 0.00009))(bn) # dropout = Dropout(0.85)(dense) # bn = BatchNormalization()(dropout) # dense = Dense(512,
# activation='softplus',kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.00009))(bn) # dropout
#  = Dropout(0.85)(dense) # bn = BatchNormalization()(dropout)
#
# output = Dense(1,activation='sigmoid')(bn) #regression
# model = Model(sequence_input,output)
# model.summary()
# # loss: 0.4232 - acc: 0.8234 - val_loss: 0.4175 - val_acc: 0.8276=>LSTM+GRU_512_2
#
#
# # In[9]:
#
#
# #regression #model/biLSTM_biGRU_reg_512.h5 # loss: 0.4008 - acc: 0.8435 - val_loss: 0.4205 - val_acc:
# 0.8363=>LSTM+GRU_512 sequence_input = Input(shape=(max_length,size)) rnn = Bidirectional(LSTM(512,
# activation='tanh',dropout=0.6,recurrent_dropout=0.55,return_sequences=True,kernel_initializer='he_uniform'))(
# sequence_input) rnn = Bidirectional(GRU(512,activation='tanh',dropout=0.7,recurrent_dropout=0.65,
# return_sequences=False,kernel_initializer='he_uniform'))(rnn) bn = BatchNormalization()(rnn)
#
# dense = Dense(1024,activation='selu',kernel_initializer='lecun_normal',kernel_regularizer=regularizers.l2(0.0009))(bn)
# dropout = Dropout(0.75)(dense)
# bn = BatchNormalization()(dropout)
# dense = Dense(1024,activation='selu',kernel_initializer='lecun_normal',kernel_regularizer=regularizers.l2(0.0009))(bn)
# dropout = Dropout(0.8)(dense)
# bn = BatchNormalization()(dropout)
#
# # dense = Dense(512,activation='softplus',kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(
# 0.00009))(bn) # dropout = Dropout(0.85)(dense) # bn = BatchNormalization()(dropout) # dense = Dense(512,
# activation='softplus',kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.00009))(bn) # dropout
#  = Dropout(0.85)(dense) # bn = BatchNormalization()(dropout)
#
# output = Dense(1,activation='sigmoid', kernel_initializer='uniform')(bn) #regression
# model = Model(sequence_input,output)
# model.summary()
#
#
# # In[11]:
#
#
# model = Sequential()
# model.add(GRU(512, dropout=0.5, recurrent_dropout=0.5, return_sequences=True, input_shape=(max_length, 256))) #39
# model.add(GRU(512, dropout=0.5, recurrent_dropout=0.5))# LSTM第一層, 1024,bigger dropout,tanh
# model.add(Dense(512, activation='selu')) #dropout, initializer
# model.add(Dense(512, activation='selu'))
# model.add(Dense(1, activation='sigmoid')) #softmax
# model.compile('adam', 'binary_crossentropy', metrics=['accuracy']) #categorical
# model.summary()
#
# model.fit(train_X, train_Y, batch_size=1024, epochs=50, validation_split=0.1)
# model.save('model/rnn.h5')
#
#
# # ## FIT
#
# # In[11]:
#
#
# #classification
# opt = adam(decay=1e-20) #decay1e-8
# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#
# batchSize=1024 #64
# patien=15
# epoch=500
# saveP = 'model/biLSTM_biLSTM_clf_1024_2.h5'
# logD = './logs/'
# history = History()
#
# callback=[
#     EarlyStopping(patience=patien,monitor='val_acc',verbose=1),
#     ModelCheckpoint(saveP,monitor='val_loss',verbose=1,save_best_only=True, save_weights_only=True),
#     TensorBoard(log_dir=logD+'events.epochs'+str(epoch)),
#     history,
# ]
# model.fit(X_train, Y_train,
#                 epochs=epoch,
#                 batch_size=batchSize,
#                 shuffle=True,
#                 validation_data=(X_valid,Y_valid),
#                 callbacks=callback,
#                 class_weight='auto'
#                 )
# model.save(saveP+"_all.h5")
# #loss: 0.3229 - acc: 0.8630 - val_loss: 0.3898 - val_acc: 0.8381=>biLSTM_biLSTM_clf_1024
# # loss: 0.3309 - acc: 0.8620 - val_loss: 0.3872 - val_acc: 0.8328=>biLSTM_biGRU_clf_1024
# # loss: 0.3745 - acc: 0.8483 - val_loss: 0.4039 - val_acc: 0.8336=>GRU_GRU
# #loss: 0.3587 - acc: 0.8422 - val_loss: 0.3661 - val_acc: 0.8385=>biLSTM_biLSTM_clf_1024_2.h5
#
#
# # In[11]:
#
#
# #regression
# model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
#
# batchSize=512
# patien=50
# epoch=500
# saveP = 'model/biLSTM_biGRU_reg_512_2.h5'
# logD = './logs/'
# history = History()
#
# callback=[
#     EarlyStopping(patience=patien,monitor='val_acc',verbose=1),
#     ModelCheckpoint(saveP,monitor='val_loss',verbose=1,save_best_only=True, save_weights_only=True),
#     TensorBoard(log_dir=logD+'events.epochs'+str(epoch)),
#     history,
# ]
# model.fit(X_train, Y_train,
#                 epochs=epoch,
#                 batch_size=batchSize,
#                 shuffle=True,
#                 validation_data=(X_valid,Y_valid),
#                 callbacks=callback,
#                 class_weight='auto'
#                 )
# model.save(saveP+"_all.h5")
# #loss: 0.3808 - acc: 0.8289 - val_loss: 0.3907 - val_acc: 0.8294=>LSTM、GRU
# # loss: 0.3994 - acc: 0.8332 - val_loss: 0.4089 - val_acc: 0.8295=>pure GRU
# #loss: 0.4142 - acc: 0.8287 - val_loss: 0.4227 - val_acc: 0.8299 =>pure LSTM
# # loss: 0.4327 - acc: 0.8191 - val_loss: 0.4292 - val_acc: 0.8285 =>GRU、LSTM
# # loss: 0.4008 - acc: 0.8435 - val_loss: 0.4205 - val_acc: 0.8363=>LSTM+GRU_512
# # loss: 0.4232 - acc: 0.8234 - val_loss: 0.4175 - val_acc: 0.8276=>LSTM+GRU_512_2
#
#
# # ## Predict
#
# # In[11]:
#
#
# model.load_weights('model/biLSTM_biGRU_reg_512.h5')
# model.predict(X_valid)
#
#
# # ***
#
# # In[12]:
#
#
# kk = model.predict(X_valid)
# kk.shape
#
#
# # In[40]:
#
#
# s = "been there ,,, still there .,... but he can ' t complain when he runs out of clean clothes i taught mine how
# to do his at 10" stemmer.stem_sentence(s)
#
#
# # In[26]:
#
#
# df = pd.read_csv('data/testing_data.txt',engine='python')
# df
#
#
# # In[19]:
#
#
# with open(train_no_label) as f:
#     train = f.readlines()
# train
#
#
# # In[20]:
#
#
# with open(test_path) as f:
#     train = f.readlines()
# train
#
