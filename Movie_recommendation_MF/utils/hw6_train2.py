
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
# import glove
# import gensim
from gensim.models.word2vec import *
from keras.preprocessing.text import *
from keras.preprocessing.sequence import *
from keras.utils import *
from keras.layers.merge import *
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.utils import *
import random
import re
from collections import Counter
from keras.utils.generic_utils import *
from keras import regularizers
import string
# import unicodedata as udata
import pickle
from keras.applications import *
from keras.preprocessing.image import *
from keras.regularizers import l2
from keras.initializers import Zeros
from keras.callbacks import History ,ModelCheckpoint, EarlyStopping


# ### hyperparameter

# In[38]:


split_rate = 0.95

type_ = 'get2'
dp=0.1
lmbda = 1e-5#1e-5
latent_dim = 64
fb_dim = 3
batchSize=2048
patien=50
epoch=500
opt=Adam(decay=1e-20,amsgrad=False)
# opt= Nadam()
# opt = Adamax(decay=1e-20)
saveP = 'model/'+type_+'_'+str(dp)+'_'+str(lmbda)+'_'+str(latent_dim)+'_'+str(fb_dim)+'_'+str(batchSize)+'.h5'
logD = './logs/'+str(latent_dim)+'_'+str(fb_dim)+'/'


# ## Prepare Data

# In[3]:


movies = pd.read_csv('data/movies.csv',sep='::',engine='python') # mid,title,type
movies.columns = ['MovieID','Title','Type']
movies['Year'] = movies['Title'].apply(lambda x:str(x)[-5:-1])
users = pd.read_csv('data/users.csv',sep='::',engine='python') # gender, age,occupationmzip
users.columns = ['UserID','Gender','Age','Occupation','Zip']
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


# In[4]:


def type2onehot(s):
    '''
    input: pandas.Series , movie type sep by |
    output: one-hot encoding list
    '''
    x = s.split('|')
    label = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary',
 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
 'War', 'Western']
    f = [0]*18
    for i,v in enumerate(label):
        if v in x:
            f[i]=1
    return f


# In[5]:


def gen2onehot(s):
    '''
    input: pandas.Series , user gender binary
    output: one-hot encoding list
    '''
    x = [s]
    label = ['F', 'M']
    f = [0]*2
    for i,v in enumerate(label):
        if v in x:
            f[i]=1
    return f


# In[6]:


def occu2onehot(s):
    '''
    input: pandas.Series , user occupation categorical
    output: one-hot encoding list
    '''
    x = [s]
    label = [10, 16, 15,  7, 20,  9,  1, 12, 17,  0,  3, 14,  4, 11,  8, 19,  2,
       18,  5, 13,  6]
    f = [0]*21
    for i,v in enumerate(label):
        if v in x:
            f[i]=1
    return f


# In[7]:


def split_valid_set(X_all, X2_all,X3_all,X4_all,Y_all, percentage=0.95):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, X2_all,X3_all,X4_all,Y_all = _shuffle(X_all, X2_all,X3_all,X4_all ,Y_all)

    X_train, X2_train,X3_train,X4_train,Y_train = X_all[0:valid_data_size],X2_all[0:valid_data_size],X3_all[0:valid_data_size] ,X4_all[0:valid_data_size],Y_all[0:valid_data_size]
    X_valid, X2_valid,X3_valid,X4_valid,Y_valid = X_all[valid_data_size:], X2_all[valid_data_size:], X3_all[valid_data_size:], X4_all[valid_data_size:],Y_all[valid_data_size:]

    return X_train,X2_train,X3_train,X4_train, Y_train, X_valid,X2_valid,X3_valid,X4_valid, Y_valid

def _shuffle(X, X2 ,X3,X4,Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
#     print(X.shape, Y.shape)
    return (X[randomize], X2[randomize],X3[randomize],X4[randomize],Y[randomize])


# In[8]:


# 重新編號
uid_encode = {}
mid_encode = {}
age_encode = {}
zip_encode = {}
year_encode = {}
for i, UserID in enumerate(np.unique(np.concatenate((train["UserID"],test["UserID"])))):
    uid_encode[UserID] = i
for j, MovieID in enumerate(np.unique(np.concatenate((train["MovieID"],test["MovieID"])))):
    mid_encode[MovieID] = j
for m , Age in enumerate(np.unique(users["Age"])):
    age_encode[Age] = m
for n, Zip in enumerate(np.unique(users['Zip'])):
    zip_encode[Zip] = n
for o , Year in enumerate(np.unique(movies['Year'])):
    year_encode[Year] = o
uid_num = len(uid_encode)
mid_num = len(mid_encode)


# In[9]:


pickle.dump(uid_encode,open('model/uid_encode.pkl','wb'))
pickle.dump(mid_encode,open('model/mid_encode.pkl','wb'))
pickle.dump(age_encode, open('model/age_encode.pkl','wb'))
pickle.dump(zip_encode, open('model/zip_encode.pkl','wb'))
pickle.dump(year_encode, open('model/year_encode.pkl','wb'))


# In[10]:


train_df = train.filter(['UserID','MovieID'])
train_df = pd.merge(train_df,users , on='UserID',how='left')
train_df = pd.merge(train_df,movies, on='MovieID',how='left')
train_df = train_df.drop(['Title'],axis=1)
train_df


# In[11]:


# dont repeat do
train_df['UserID'] = train_df.UserID.map(uid_encode)
train_df['MovieID'] = train_df.MovieID.map(mid_encode)
train_df['Age'] = train_df.Age.map(age_encode)
train_df['Zip'] = train_df.Zip.map(zip_encode)
train_df['Year'] = train_df.Year.map(year_encode)

train_df['Gender'] = train_df.Gender.map(gen2onehot)
train_df['Occupation'] = train_df.Occupation.map(occu2onehot)
train_df['Type'] = train_df.Type.map(type2onehot)
train_df


# In[12]:


train_df_user = train_df.filter(['UserID'])

train_df_user_fb = train_df.filter(['Age','Zip'])
temp = pd.DataFrame(train_df['Gender'].values.tolist())
train_df_user_fb = pd.merge(temp,train_df_user_fb,how='right',right_index=True,left_index=True)
temp = pd.DataFrame(train_df['Occupation'].values.tolist())
train_df_user_fb = pd.merge(train_df_user_fb,temp,how='left',right_index=True,left_index=True)

train_df_movie = train_df.filter(['MovieID'])

train_df_movie_fb = train_df.filter(['Year'])
temp = pd.DataFrame(train_df['Type'].values.tolist())
train_df_movie_fb = pd.merge(train_df_movie_fb,temp,how='left',right_index=True,left_index=True)


# In[13]:


train_user = np.array(train_df_user)
train_user_fb = np.array(train_df_user_fb)
train_movie = np.array(train_df_movie)
train_movie_fb = np.array(train_df_movie_fb)
train_rating = np.array(train["Rating"])
print(train_user.shape , train_user_fb.shape, train_movie.shape, train_movie_fb.shape , train_rating.shape)


# In[14]:


rating_mean = np.mean(train_rating)
rating_std = np.std(train_rating)
train_rating = (train_rating-rating_mean) / rating_std
#     train_user = (train_user-np.mean(train_user)) / np.std(train_user)
#     train_movie = (train_movie-np.mean(train_movie)) / np.std(train_movie)
print(rating_mean,rating_std)
def rmse(y_true, y_pred):
    y_true = y_true*rating_std+rating_mean
    y_pred = y_pred*rating_std+rating_mean
    y_pred = K.clip(y_pred, 1.0, 5.0) #rating range
    return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))
train_user,train_user_fb ,train_movie,train_movie_fb,train_rating, valid_user,valid_user_fb, valid_movie,valid_movie_fb,valid_rating = split_valid_set(train_user,train_user_fb,train_movie,train_movie_fb ,train_rating, 0.9)


# In[15]:


print(train_user.shape , train_user_fb.shape, train_movie.shape, train_movie_fb.shape , train_rating.shape)


# ## Model

# In[16]:


def get_model(n_users, n_items,dp=0.5,lmbda=1e-4,latent_dim=256,fb_dim=256):
    get_custom_objects().update({"rmse": rmse})
    
    user_input = Input(shape=[1])
    user_fb_input = Input(shape=[25])
    item_input = Input(shape=[1])
    item_fb_input = Input(shape=[19])
    
    user_vec = Embedding(n_users, latent_dim , embeddings_initializer='random_normal'
                         ,embeddings_regularizer=l2(lmbda))(user_input)
    user_vec = Flatten()(user_vec)
    
    user_fb_vec = Embedding(input_dim=n_users , output_dim=fb_dim , embeddings_initializer='lecun_normal', 
                            embeddings_regularizer=l2(lmbda))(user_fb_input)
#     user_fb_vec = Dropout(0.1)(user_fb_vec)
    user_fb_vec = Flatten()(user_fb_vec)
    
    item_vec = Embedding(n_items, latent_dim, embeddings_initializer='lecun_normal'
                         ,embeddings_regularizer=l2(lmbda))(item_input)
    item_vec = Flatten()(item_vec)
    
    item_fb_vec = Embedding(input_dim=n_items, output_dim=int(fb_dim*0.76), embeddings_initializer='lecun_normal',
                            embeddings_regularizer=l2(lmbda))(item_fb_input)
#     item_fb_vec = Dropout(0.1)(item_fb_vec)
    item_fb_vec = Flatten()(item_fb_vec)
    
    user_fb_dense = BatchNormalization()(user_fb_vec)
    user_fb_dense = Dense(64,activation='selu',kernel_initializer='lecun_normal'
                          ,kernel_regularizer=l2(lmbda*2))(user_fb_dense)
    user_fb_dense = Dropout(dp)(user_fb_dense)
    user_fb_dense = BatchNormalization()(user_fb_dense)
    user_fb_dense = Dense(32,activation='selu',kernel_initializer='lecun_normal'
                          ,kernel_regularizer=l2(lmbda*2))(user_fb_dense)
    user_fb_dense = Dropout(dp)(user_fb_dense)
    user_fb_dense = BatchNormalization()(user_fb_dense)
    user_fb_dense = Dense(16,activation='selu',kernel_initializer='lecun_normal'
                          ,kernel_regularizer=l2(lmbda*2))(user_fb_dense)
    user_fb_dense = Dropout(dp)(user_fb_dense)
    user_fb_dense = BatchNormalization()(user_fb_dense)
    user_fb_dense = Dense(8,activation='selu',kernel_initializer='lecun_normal'
                          ,kernel_regularizer=l2(lmbda*2))(user_fb_dense)
    user_fb_dense = Dropout(dp)(user_fb_dense)
    user_fb_dense = BatchNormalization()(user_fb_dense)

    movie_fb_dense = BatchNormalization()(item_fb_vec)
    movie_fb_dense = Dense(32,activation='selu',kernel_initializer='lecun_normal'
                          ,kernel_regularizer=l2(lmbda*2))(movie_fb_dense)
    movie_fb_dense = Dropout(dp)(movie_fb_dense)
    movie_fb_dense = BatchNormalization()(movie_fb_dense)
    movie_fb_dense = Dense(16,activation='selu',kernel_initializer='lecun_normal'
                          ,kernel_regularizer=l2(lmbda*2))(movie_fb_dense)
    movie_fb_dense = Dropout(dp)(movie_fb_dense)
    movie_fb_dense = BatchNormalization()(movie_fb_dense)
    movie_fb_dense = Dense(16,activation='selu',kernel_initializer='lecun_normal'
                          ,kernel_regularizer=l2(lmbda*2))(movie_fb_dense)
    movie_fb_dense = Dropout(dp)(movie_fb_dense)
    movie_fb_dense = BatchNormalization()(movie_fb_dense)
    movie_fb_dense = Dense(8,activation='selu',kernel_initializer='lecun_normal'
                          ,kernel_regularizer=l2(lmbda*2))(movie_fb_dense)
    movie_fb_dense = Dropout(dp)(movie_fb_dense)
    movie_fb_dense = BatchNormalization()(movie_fb_dense)
    
    user_con = Concatenate()([user_vec,user_fb_dense])
    movie_con = Concatenate()([item_vec, movie_fb_dense])
    
    r_hat = dot([user_con,movie_con],axes=1)
    
    user_con_ori = Concatenate()([user_input,user_fb_input])
    movie_con_ori = Concatenate()([item_input,item_fb_input])
    
    user_bias = Embedding(n_users,1, embeddings_initializer="zeros",
                          embeddings_regularizer=l2(lmbda))(user_con_ori)
    user_bias = Flatten()(user_bias)
    user_bias = Dense(1,activation='linear')(user_bias)
    item_bias = Embedding(n_items, 1, embeddings_initializer="zeros",
                          embeddings_regularizer=l2(lmbda))(movie_con_ori)
    item_bias = Flatten()(item_bias)
    item_bias = Dense(1,activation='linear')(item_bias)
    r_hat = add([r_hat, user_bias, item_bias])
    model = Model([user_input,user_fb_input,item_input,item_fb_input],r_hat)
    model.summary()
    return model


# In[39]:


def get_model2(n_users, n_items,dp=0.5,lmbda=1e-4,latent_dim=256):
    get_custom_objects().update({"rmse": rmse})
    
    user_input = Input(shape=[1])
    user_fb_input = Input(shape=[25])
    item_input = Input(shape=[1])
    item_fb_input = Input(shape=[19])
    
    user_vec = Embedding(n_users, latent_dim , embeddings_initializer='lecun_normal'
                         ,embeddings_regularizer=l2(lmbda))(user_input)
    user_vec = Flatten()(user_vec)
    user_conca = Concatenate()([user_vec,user_fb_input])
    
    
    item_vec = Embedding(n_items, int(latent_dim/2), embeddings_initializer='lecun_normal'
                         ,embeddings_regularizer=l2(lmbda))(item_input)
    item_vec = Flatten()(item_vec)
    movie_conca = Concatenate()([item_vec,item_fb_input])
    
    
    user_dense = BatchNormalization()(user_conca)
    user_dense = Dense(latent_dim,activation='selu',kernel_initializer='lecun_normal'
                          ,kernel_regularizer=l2(lmbda*2))(user_dense)
    user_dense = Dropout(dp)(user_dense)
    user_dense = BatchNormalization()(user_dense)
    user_dense = Dense(int(latent_dim/2),activation='selu',kernel_initializer='lecun_normal'
                          ,kernel_regularizer=l2(lmbda*2))(user_dense)
    user_dense = Dropout(dp)(user_dense)
    user_dense = BatchNormalization()(user_dense)
    user_dense = Dense(int(latent_dim/4),activation='selu',kernel_initializer='lecun_normal'
                          ,kernel_regularizer=l2(lmbda*2))(user_dense)
    user_dense = Dropout(dp)(user_dense)
    user_dense = BatchNormalization()(user_dense)
    user_dense = Dense(int(latent_dim/16),activation='selu',kernel_initializer='lecun_normal'
                          ,kernel_regularizer=l2(lmbda*2))(user_dense)
    user_dense = Dropout(dp)(user_dense)
#     user_dense = BatchNormalization()(user_dense)

    
    movie_dense = BatchNormalization()(movie_conca)
    movie_dense = Dense(int(latent_dim/2),activation='selu',kernel_initializer='lecun_normal'
                          ,kernel_regularizer=l2(lmbda*2))(movie_dense)
    movie_dense = Dropout(dp)(movie_dense)
    movie_dense = BatchNormalization()(movie_dense)
    movie_dense = Dense(int(latent_dim/4),activation='selu',kernel_initializer='lecun_normal'
                          ,kernel_regularizer=l2(lmbda*2))(movie_dense)
    movie_dense = Dropout(dp)(movie_dense)
    movie_dense = BatchNormalization()(movie_dense)
    movie_dense = Dense(int(latent_dim/4),activation='selu',kernel_initializer='lecun_normal'
                          ,kernel_regularizer=l2(lmbda*2))(movie_dense)
    movie_dense = Dropout(dp)(movie_dense)
    movie_dense = BatchNormalization()(movie_dense)
    movie_dense = Dense(int(latent_dim/16),activation='selu',kernel_initializer='lecun_normal'
                          ,kernel_regularizer=l2(lmbda*2))(movie_dense)
    movie_dense = Dropout(dp)(movie_dense)
#     movie_dense = BatchNormalization()(movie_dense)
    
#     user_con = Concatenate()([user_vec,user_fb_dense])
#     movie_con = Concatenate()([item_vec, movie_fb_dense])
    
    r_hat = Dot(axes=-1)([user_dense,movie_dense])
    
    user_con_ori = Concatenate()([user_input,user_fb_input])
    movie_con_ori = Concatenate()([item_input,item_fb_input])
    
    user_bias = Embedding(n_users,1, embeddings_initializer="lecun_normal",
                          embeddings_regularizer=l2(lmbda))(user_con_ori)
    user_bias = Flatten()(user_bias)
#     user_bias = BatchNormalization()(user_con_ori)
#     user_bias = Dense(26,activation='selu',kernel_initializer='lecun_normal',
#                       kernel_regularizer=l2(lmbda*3))(user_bias)
    user_bias = BatchNormalization()(user_bias)
    user_bias = Dense(1,activation='selu',kernel_initializer='lecun_normal',
                      kernel_regularizer=l2(lmbda*3))(user_bias)
#     user_bias = BatchNormalization()(user_bias)
    user_bias = Dense(1,activation='linear',kernel_initializer='lecun_normal')(user_bias)
#     user_bias = BatchNormalization()(user_bias)
    
    movie_bias = Embedding(n_items, 1, embeddings_initializer="lecun_normal",
                          embeddings_regularizer=l2(lmbda))(movie_con_ori)
    movie_bias = Flatten()(movie_bias)    
#     movie_bias = BatchNormalization()(movie_con_ori)
#     movie_bias = Dense(20,activation='selu',kernel_initializer='lecun_normal',
#                       kernel_regularizer=l2(lmbda*3))(movie_bias)
    movie_bias = BatchNormalization()(movie_bias)
    movie_bias = Dense(1,activation='selu',kernel_initializer='lecun_normal',
                      kernel_regularizer=l2(lmbda*3))(movie_bias)
#     movie_bias = BatchNormalization()(movie_bias)
    movie_bias = Dense(1,activation='linear',kernel_initializer='lecun_normal')(movie_bias)
#     movie_bias = BatchNormalization()(movie_bias)
    
    r_hat = add([r_hat, user_bias, movie_bias])
    model = Model([user_input,user_fb_input,item_input,item_fb_input],r_hat)
    model.summary()
    return model


# In[40]:


if type_ == 'get2':
    print('getmodel2')
    model = get_model2(uid_num,mid_num,dp=dp,lmbda=lmbda,latent_dim=latent_dim)
else:
    model = get_model(uid_num,mid_num,dp=dp,lmbda=lmbda,latent_dim=latent_dim,fb_dim=fb_dim)
model.compile(optimizer=opt, loss='mse', metrics=[rmse])


# In[41]:




history = History()
callback=[
    EarlyStopping(patience=patien,monitor='val_loss',verbose=1),
    ModelCheckpoint(saveP,monitor='val_rmse',verbose=1,save_best_only=True, save_weights_only=True),
    TensorBoard(log_dir=logD+saveP),
    history,
]
model.fit([train_user,train_user_fb,train_movie,train_movie_fb], train_rating,
                epochs=epoch,
                batch_size=batchSize,
                shuffle=True,
                validation_data=([valid_user,valid_user_fb,valid_movie,valid_movie_fb],valid_rating),
                callbacks=callback, 
                class_weight='auto'
                )
model.save(saveP+"_all.h5")
# loss: 0.5472 - rmse: 0.7925 - val_loss: 0.6268 - val_rmse: 0.8517


# In[60]:


# kk=[]
# kk.append(model)
# print(kk[0])
# pickle.dump(kk,open('model/modeltest.pkl','wb'),protocol=-1)


# In[94]:


# pd.DataFrame(train_df['Occupation'].values.tolist())
# train_df['Occupation']


# In[ ]:


users['Zip-code']


# In[38]:


# len(movies['Type'].unique())
types = movies['Type'].str.cat(sep='|')
set(types.split('|'))


# In[46]:


# movies['Type'].map(type2onehot)[0]


# In[49]:


# len(movies['Year'].unique())


# In[66]:


# users['Gender'].map(gen2onehot)


# In[57]:


# label = [10, 16, 15,  7, 20,  9,  1, 12, 17,  0,  3, 14,  4, 11,  8, 19,  2,
#    18,  5, 13,  6]

