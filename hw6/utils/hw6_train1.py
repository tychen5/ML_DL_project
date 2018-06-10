
# coding: utf-8

# In[92]:


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

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style='darkgrid', rc={'figure.facecolor':'white'}, font_scale=1.2)

K.set_image_dim_ordering('tf')


# from utils import feature, utils


# ## Data prepare

# In[93]:


norm_std = True #Z
norm_range = False #M
split_rate = 0.7

bias = False
latent_dim = 128
batchSize=1024
patien=3
epoch=350
saveP = 'model/'+str(norm_std)+'_'+str(norm_range)+'_'+str(bias)+'_'+str(latent_dim)+'_'+str(batchSize)+'.h5'
logD = './logs/'
opt=Adam(decay=1e-20,amsgrad=False)

movies = pd.read_csv('data/movies.csv',sep='::',engine='python') # mid,title,type
movies.columns = ['MovieID','Title','Type']
users = pd.read_csv('data/users.csv',sep='::',engine='python') # gender, age,occupationmzip
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


# In[94]:


def split_valid_set(X_all, X2_all,Y_all, percentage=0.8):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, X2_all,Y_all = _shuffle(X_all, X2_all ,Y_all)

    X_train, X2_train,Y_train = X_all[0:valid_data_size],X2_all[0:valid_data_size] ,Y_all[0:valid_data_size]
    X_valid, X2_valid,Y_valid = X_all[valid_data_size:], X2_all[valid_data_size:],Y_all[valid_data_size:]

    return X_train,X2_train, Y_train, X_valid,X2_valid, Y_valid

def _shuffle(X, X2 ,Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
#     print(X.shape, Y.shape)
    return (X[randomize], X2[randomize],Y[randomize])


# In[95]:



# 重新編號
uid_encode = {}
mid_encode = {}
for i, UserID in enumerate(np.unique(np.concatenate((train["UserID"],test["UserID"])))):
    uid_encode[UserID] = i
for j, MovieID in enumerate(np.unique(np.concatenate((train["MovieID"],test["MovieID"])))):
    mid_encode[MovieID] = j
uid_num = len(uid_encode)
mid_num = len(mid_encode)
print('uid#',uid_num,'mid#',mid_num)


# In[96]:


pickle.dump(uid_encode,open('model/uid_encode.pkl','wb'))
pickle.dump(mid_encode,open('model/mid_encode.pkl','wb'))


# In[97]:


train_user = np.array([ uid_encode[i] for i in train["UserID"] ])
train_movie = np.array([ mid_encode[j] for j in train["MovieID"] ])
train_rating = np.array(train["Rating"])
# len(train_rating)


# In[98]:



if norm_std:
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
    
elif norm_range:
    rating_range = np.max(train_rating)-np.min(train_rating)
    rating_min = np.min(train_rating)
    train_rating = (train_rating-rating_min) / (rating_range)
#     train_user = (train_user-np.min(train_user)) / (np.max(train_user)-np.min(train_user)) 
#     train_movie = (train_movie-np.min(train_movie)) / (np.max(train_movie)-np.min(train_movie)) 
    print(rating_min,rating_range)
    def rmse(y_true, y_pred):
        y_true = y_true*rating_range+rating_min
        y_pred = y_pred*rating_range+rating_min
        y_pred = K.clip(y_pred, 1.0, 5.0) #限制在1~5
        return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))
else:
    def rmse(y_true, y_pred):
        y_pred = K.clip(y_pred, 1.0, 5.0) #限制在1~5
        return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))


train_user, train_movie,train_rating, valid_user, valid_movie,valid_rating = split_valid_set(train_user,train_movie ,train_rating, split_rate)
print(train_user.shape , train_movie.shape , train_rating.shape)


# In[ ]:


# movies['MovieID'] = movies['MovieID'].astype('int')
# key
# movie_emb[genres_map[key]]
# key
# print(genres_map[key])
# genres_map[key] = movie_emb[genres_map[key]]
# print(movies['movieID'])
# clean_dict = filter(lambda k: not isnan(genres_map[k]), genres_map)
# clean_dict = {k: genres_map[k] for k in genres_map if not isnan(genres_map[k])}
# clean_dict


# ## Model

# In[99]:


def get_model(n_users, n_items, bias=True,latent_dim=256):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    user_vec = Embedding(n_users, latent_dim , embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    r_hat = Dot(axes=1)([user_vec,item_vec])
    user_bias = Embedding(n_users,1, embeddings_initializer="zeros")(user_input)
    user_bias = Flatten()(user_bias)
    item_bias = Embedding(n_items, 1, embeddings_initializer="zeros")(item_input)
    item_bias = Flatten()(item_bias)
    r_hat = Add()([r_hat, user_bias, item_bias])
    model = Model([user_input,item_input],r_hat)
    model.summary()
    return model


# ## Fit

# In[100]:




model = get_model(uid_num,mid_num,bias,latent_dim)
model.compile(optimizer=opt, loss='mse', metrics=[rmse])
history = History()

   
    


# In[101]:


callback=[
EarlyStopping(patience=patien,monitor='val_loss',verbose=1),
ModelCheckpoint(saveP,monitor='val_rmse',verbose=1,save_best_only=True, save_weights_only=True),
TensorBoard(log_dir=logD+'non_bias'),
history,
]
model.fit([train_user,train_movie], train_rating,
            epochs=epoch,
            batch_size=batchSize,
            shuffle=True,
            validation_data=([valid_user,valid_movie],valid_rating),
            callbacks=callback, 
            class_weight='auto'
            )
model.save(saveP+"_all.h5")
#   loss: 0.6746 - rmse: 0.8195 - val_loss: 0.7638 - val_rmse: 0.8718
# loss: 0.5187 - rmse: 0.8030 - val_loss: 0.6062 - val_rmse: 0.8683 #model/True_False_True_128_256.h5
#  loss: 0.0397 - rmse: 0.7951 - val_loss: 0.0508 - val_rmse: 0.8991 # model/False_True_True_128_256.h5


# ## Report

# In[ ]:


from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from math import isnan


# In[ ]:


max_userid = uid_num 
max_movieid = mid_num 
DIM = 128
TEST_CSV = 'data/test.csv'
USERS_CSV = 'data/users.csv'
MOVIES_CSV = 'data/movies.csv'
classes = ["Action|Romance", "Animation|Children's", "Drama|Musical", "War|Documentary",'Comedy','Horror|Thriller|Mystery']


# In[ ]:



def draw(mapping, filename):
    print('Drawing...')
    fig = plt.figure(figsize=(10, 10), dpi=200)
    for i, key in enumerate(mapping.keys()):
        vis_x = mapping[key][:, 0]
        vis_y = mapping[key][:, 1]
        plt.scatter(vis_x, vis_y, marker='.', label=key)
    plt.xticks([])
    plt.yticks([])
    plt.legend(scatterpoints=1,
               loc='lower left',
               fontsize=8)
    plt.tight_layout()
    # plt.show()
    fig.savefig(filename)
    print('Done drawing!')


# In[ ]:


users = pd.read_csv(USERS_CSV, sep='::', engine='python',
        usecols=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
print('{} description of {} users loaded'.format(len(users), max_userid))

# movies = pd.read_csv(MOVIES_CSV, sep='::', engine='python',
#         usecols=['movieID', 'Title', 'Genres'])
movies = pd.read_csv('data/movies.csv',sep='::',engine='python')
movies.columns = ['MovieID','Title','Type']
print('{} descriptions of {} movies loaded'.format(len(movies), max_movieid))

users['UserID'] = users.UserID.map(uid_encode).astype('int')
movies['MovieID'] = movies.MovieID.map(mid_encode)
# movies = movies.fillna(0)
movies = movies.dropna()
movies['MovieID'] = movies['MovieID'].astype('int')

test_data = pd.read_csv(TEST_CSV, usecols=['UserID', 'MovieID'])
test_data['UserID'] = test_data.UserID.map(uid_encode).astype('int')
test_data['MovieID'] = test_data.MovieID.map(mid_encode).astype('int')
print('{} testing data loaded.'.format(test_data.shape[0]))

trained_model = get_model(uid_num,mid_num,bias,latent_dim)
print('Loading model weights...')
trained_model.load_weights('models/True_False_True_128_256.h5')
print('Loading model done!!!')

movies_array = movies.as_matrix()
genres_map = {}
for i in range(movies_array.shape[0]):
    genre = movies_array[i][2].split('|')[0]
    if genre not in genres_map.keys():
        genres_map[genre] = [movies_array[i][0] - 1]
    else:
        genres_map[genre].append(movies_array[i][0] - 1)
# print(genres_map)
movie_emb = np.array(trained_model.layers[3].get_weights()).squeeze()
model = TSNE(n_components=2, random_state=0)
movie_emb = model.fit_transform(movie_emb)
# print(key, genres_map[key])
for key in genres_map.keys():
    genres_map[key] = movie_emb[genres_map[key]]

new_genres_map = {}
for c in classes:
    new_genres_map[c] = np.ndarray(shape=(0, 2))
    for g in c.split('|'):
        new_genres_map[c] = np.concatenate((new_genres_map[c], genres_map[g]), axis=0)
#     print(new_genres_map[c].shape)
draw(new_genres_map, 'graph.png')

