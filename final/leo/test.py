
# coding: utf-8

# In[241]:


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
from collections import Counter
from keras import *
from keras.engine.topology import *
from keras.optimizers import *
import keras
# import pandas as pd
import glob
from sklearn.semi_supervised import *
import pickle
from keras.applications import *
from keras.preprocessing.image import *
from keras.losses import mse, binary_crossentropy
import pandas as pd # data frame
import numpy as np # matrix math
from scipy.io import wavfile # reading the wavfile
from sklearn.utils import shuffle # shuffling of data
from random import sample # random selection
from tqdm import tqdm # progress bar
import matplotlib.pyplot as plt # to view graphs
import wave
from math import log, floor
# audio processing
from scipy import signal # audio processing
from scipy.fftpack import dct
import librosa # library for audio processing
import numpy as np
import pandas as pd
from sklearn.decomposition import *
from sklearn.cluster import KMeans
import sys, os
import random,math
from tqdm import tqdm ##
# from xgboost.sklearn import XGBClassifier
from sklearn.utils import shuffle # shuffling of data
from random import sample # random selection
from tqdm import tqdm # progress bar
# audio processing
from scipy import signal # audio processing
from scipy.fftpack import dct
import librosa # library for audio processing
# import xgboost as xgb
# import lightgbm as lgb
# import catboost as ctb
from keras.utils import *
from sklearn.ensemble import *
import pickle
# from bayes_opt import BayesianOptimization
from logHandler import Logger
from utils import readCSV, getPath, writePickle,readPickle
from keras.regularizers import l2
from keras.callbacks import History ,ModelCheckpoint, EarlyStopping

import resnet
from random_eraser import get_random_eraser
from mixup_generator import MixupGenerator


# In[242]:


predict_path = sys.argv[1]


# ## voting 3. ensemble model predictions

# In[243]:


# type_ = 'mfcc7' #要抽取哪一個種類的unverified trainX出來去re-train
un_or_test = 'combine' # unverified or test
phase = 'phase4'

folder = 'leo/data/'+phase+'/'+un_or_test+'/' #共同predict對unverified data的結果



acc_df = pd.read_csv('leo/data/'+phase+'/weight_accF.csv') # acc csv檔名格式: (csv,acc)
# acc_df.columns = ['unverified','test','acc']
acc_df.columns = [un_or_test,'acc']
acc_df = acc_df.filter([un_or_test,'acc'])
files = os.listdir(folder)

ratio_all=0
for i,csv in enumerate(files):
    if csv.startswith('valid_acc'):
        continue
    else:
        ratio = acc_df[acc_df[un_or_test] == csv]['acc'].values[0]
#         print(ratio)
        ratio_all += ratio
    df = pd.read_csv(os.path.join(folder,csv)) #ori method
#     df = pd.read_csv(os.path.join(folder,csv),header=0,index_col=0) # new method
#     df.sort_values("fname", inplace=True)
    if df.iloc[0,0] == 'fname':
        df = df.drop(0,axis=0)
#     df = df.drop(0,axis=1) #ori method
    df = df.drop(['fname'],axis=1) #mew mthod

    if i==0:
        train_X = df.values*ratio#+4e-4
    else:
#         try:
#         train_X *= df.values**ratio+4e-4
        train_X += df.values*ratio#+1e-3
#         except:
#             train_X += df.values[5763:]*ratio
print(train_X.shape)
# train_X = train_X ** (1/ratio_all)
train_X = train_X /ratio_all
print(sum(train_X[0]),sum(train_X[1000]),sum(train_X[2000]))
print(sum(sum(train_X))/9400)
reverse_dict = pickle.load(open('leo/data/map_reverse.pkl' , 'rb'))


# In[244]:


fname_test = pd.read_csv('leo/data/sample_submission.csv')
fname_test['label'] = 'none'
fname = fname_test # 記得註解掉如果是un+test
'''
fname_un = pd.read_csv('data/train_label.csv')
fname_un = fname_un[fname_un.manually_verified==0]
fname_un = fname_un.drop(['manually_verified'],axis=1)
fname = fname_un.append(fname_test)
fname.sort_values('fname',inplace=True)
fname.reset_index(drop=True,inplace=True)
'''
fname['label_verified'] = "none"
fname['verified_confidence']=0.0
print(len(fname))
for i,r in fname.iterrows():
    top3 = train_X[i].argsort()[-3:][::-1]
    result = [reverse_dict[x] for x in top3]
    s = ' '.join(result)
    fname.iloc[i,2] = s#np.argmax(train_X[i])
    fname.iloc[i,3] = max(train_X[i])

df = fname
print(df['verified_confidence'].min(),df['verified_confidence'].mean(), df['verified_confidence'].std() )
# (df)


# In[245]:


df_fin = df[df.label=='none']
df_fin = df_fin.filter(['fname','label_verified'])
df_fin.columns = ['fname','label']
# df_fin.to_csv(predict_path,index=False)
# df_fin


# In[246]:


df_ens = pd.DataFrame(df_fin.label.str.split(' ',2).tolist(),columns=['1','2','3'])
df_ens = pd.merge(pd.DataFrame(df_fin.fname),df_ens,how='inner',right_index=True,left_index=True)
df_ens['lp']='none'
# df_ens


# ## stacking stage2 prediction

# In[247]:


folder = 'leo/data/stacking/lp_model_res/' 
files = os.listdir(folder)
# print(files)


# In[248]:


df_un_ans = pd.DataFrame()
for fold in files:
    un_ans = np.load(folder+fold)
    df = pd.DataFrame(un_ans).T
    df_un_ans = df_un_ans.append(df)
col_list = []
for col_num in range(len(df_un_ans.columns)):

    counter = df_un_ans[col_num].value_counts()

    col_list.append(dict(counter))

print(len(col_list) )


# In[249]:


take_list=[]
take_label=[]
for i,stats in enumerate(col_list):
    if max(stats.values()) >=len(files):
        take_list.append(i)
        ens_label = max(stats.keys(), key=(lambda k: stats[k]))
        take_label.append(ens_label)
print(len(take_list))


# In[250]:


for i,row in enumerate(take_list):
    df_ens.iloc[row,4]=take_label[i]
df_ens['lp']=df_ens.lp.map(reverse_dict)
# df_ens


# ## stacking stage2 model prediction

# In[251]:


folder = 'leo/data/stacking/nn/' 
files = os.listdir(folder)
# print(files)


# In[252]:


un_or_test = 'stack'
acc_df = pd.read_csv('leo/data/stacking/stack_accF.csv') # acc csv檔名格式: (csv,acc)
# acc_df.columns = ['unverified','test','acc']
acc_df.columns = [un_or_test,'acc']
# acc_df = acc_df.filter([un_or_test,'acc'])

ratio_all=0
for i,csv in enumerate(files):
    if csv.startswith('valid_acc'):
        continue
    else:
        ratio = acc_df[acc_df[un_or_test] == csv]['acc'].values[0]
#         print(ratio)
        ratio_all += ratio
    df = np.load(folder+csv)#pd.read_csv(os.path.join(folder,csv)) #ori method
#     df = pd.read_csv(os.path.join(folder,csv),header=0,index_col=0) # new method
#     df.sort_values("fname", inplace=True)
#     if df.iloc[0,0] == 'fname':
#         df = df.drop(0,axis=0)
#     df = df.drop(0,axis=1) #ori method
#     df = df.drop(['fname'],axis=1) #mew mthod

    if i==0:
        train_X = df*ratio #+ 8e-3
    else:
        train_X += df*ratio #+ 1e-2
#         train_X *= df**ratio + 8e-3
#         except:
#             train_X += df.values[5763:]*ratio
print(train_X.shape)
# train_X = train_X ** (1/ratio_all)
train_X = train_X /ratio_all
print(sum(train_X[0]),sum(train_X[500]),sum(train_X[2500]))
print(sum(sum(train_X))/9400)
reverse_dict = pickle.load(open('leo/data/map_reverse.pkl' , 'rb'))


# In[253]:


fname_test = pd.read_csv('leo/data/sample_submission.csv')
fname_test['label'] = 'none'
for i,r in fname_test.iterrows():
    top3 = train_X[i].argsort()[-1:][::-1]
    result = [reverse_dict[x] for x in top3]
    s = ' '.join(result)
    fname_test.iloc[i,1] = s#np.argmax(train_X[i])
#     fname.iloc[i,3] = max(train_X[i])
fname_test.columns = ['fname','stack']
# fname_test


# In[254]:


df_all = pd.merge(df_ens,fname_test,how='inner',on='fname')
# df_all = pd.read_csv('result/df_all.csv')
df_all.fillna('none',inplace=True)
print(df_all)


# In[255]:


def one(x):
    if x['lp'] != 'none':
        return x['lp']
    else:
        return x['stack']
def two(x):
    if x['fin1'] == x['stack'] == x['1']:
        return x['2']
    elif x['fin1'] == x['stack']:
        return x['1']
    else:
        return x['stack']
def three(x):
    if x['fin2'] == x['1']:
        return x['2']
    elif x['fin2'] == x['2']:
        return x['3']
    else:
        return x['1']


# In[256]:


df_all['fin1'] = df_all.apply(one,axis=1)
df_all['fin2'] = df_all.apply(two,axis=1)
df_all['fin3'] = df_all.apply(three,axis=1)
df_all['final'] = df_all['fin1']+' '+df_all['fin2']+' '+df_all['fin3']
# df_all


# In[257]:


dfF = df_all.filter(['fname','final'])
dfF.columns = ['fname','label']
print(dfF)


# In[258]:


dfF.to_csv(predict_path,index=False)
print(predict_path)

