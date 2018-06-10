
# coding: utf-8

# In[1]:


try:
    import os,sys
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
    from keras.preprocessing.text import *
    from keras.preprocessing.sequence import *
    from keras.utils import *
    from keras.layers.merge import *
    from sklearn.model_selection import *
    from sklearn.preprocessing import *
    from sklearn.utils import *
    import random
    from keras.utils.generic_utils import *
    from keras import regularizers
    import pickle
    import numpy as np
    from random import shuffle
    from math import log, floor
    import pandas as pd
except:
    pass


# In[2]:


test = pd.read_csv(sys.argv[1])#('data/test.csv')
predict_path = sys.argv[2]#'result/res.csv'
uid_encode = pickle.load(open('models/uid_encode.pkl','rb'))
mid_encode = pickle.load(open('models/mid_encode.pkl','rb'))

norm_std = True
norm_range = False
bias = True
latent_dim = 128


# In[3]:


test_user = np.array([ uid_encode[i] for i in test["UserID"] ])
test_movie = np.array([ mid_encode[j] for j in test["MovieID"] ])
uid_num = len(uid_encode)
mid_num = len(mid_encode)


# In[4]:


if norm_std:
    rating_mean = 3.5817120860388076
    rating_std = 1.116897661146206
    def rmse(y_true, y_pred):
        y_true = y_true*rating_std+rating_mean
        y_pred = y_pred*rating_std+rating_mean
        y_pred = K.clip(y_pred, 1.0, 5.0) #rating range
        return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))
    
elif norm_range:  
    rating_min = 1
    rating_range = 4
    def rmse(y_true, y_pred):
        y_true = y_true*rating_range+rating_min
        y_pred = y_pred*rating_range+rating_min
        y_pred = K.clip(y_pred, 1.0, 5.0) #限制在1~5
        return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))
print(test_user.shape , test_movie.shape)


# ## Model

# In[5]:


def get_model(n_users, n_items, bias=True,latent_dim=256):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    user_vec = Embedding(n_users, latent_dim , embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    r_hat = Dot(axes=1)([user_vec,item_vec])
    if bias:
        user_bias = Embedding(n_users,1, embeddings_initializer="zeros")(user_input)
        user_bias = Flatten()(user_bias)
        item_bias = Embedding(n_items, 1, embeddings_initializer="zeros")(item_input)
        item_bias = Flatten()(item_bias)
        r_hat = Add()([r_hat, user_bias, item_bias])
        print('=using bias=')
    model = Model([user_input,item_input],r_hat)
    model.summary()
    return model


# In[6]:


model = get_model(uid_num,mid_num,bias,latent_dim)
model.load_weights('models/True_False_True_128_256.h5') #public:0.87208 private:0.86541


# In[7]:


ans = model.predict([test_user,test_movie])
if norm_std:
    ans = ans*rating_std+rating_mean
elif norm_range:
    ans = ans*rating_range+rating_min
ans = np.clip(ans,1,5) # 控制於0~5之間


# In[8]:


# df['id']=df.index+1
df = pd.DataFrame(ans,columns=['Rating'])
df['TestDataID']=df.index +1
df = df[['TestDataID','Rating']]
df.to_csv(predict_path,index=False)
# df
print(predict_path)


# ## Report

# In[32]:


# max_userid = uid_num 
# max_movieid = mid_num 
# DIM = 128
# TEST_CSV = 'data/test.csv'
# USERS_CSV = 'data/users.csv'
# MOVIES_CSV = 'data/movies.csv'
# classes = ["Action|Romance", "Animation|Children's", "Drama|Musical", "War|Documentary",'Comedy','Horror|Thriller|Mystery']


# In[11]:


# def ensure_dir(file_path):
#   directory = os.path.dirname(file_path)
#   if len(directory) == 0: return
#   if not os.path.exists(directory):
#     os.makedirs(directory)


# In[12]:


# def parse_args():
#     parser = argparse.ArgumentParser(description='HW6: drawing graph')
#     parser.add_argument('data_dir', type=str)
#     return parser.parse_args()

# def draw(mapping, filename):
#     print('Drawing...')
#     fig = plt.figure(figsize=(10, 10), dpi=200)
#     for i, key in enumerate(mapping.keys()):
#         vis_x = mapping[key][:, 0]
#         vis_y = mapping[key][:, 1]
#         plt.scatter(vis_x, vis_y, marker='.', label=key)
#     plt.xticks([])
#     plt.yticks([])
#     plt.legend(scatterpoints=1,
#                loc='lower left',
#                fontsize=8)
#     plt.tight_layout()
#     # plt.show()
#     fig.savefig(filename)
#     print('Done drawing!')


# In[33]:


# users = pd.read_csv(USERS_CSV, sep='::', engine='python',
#         usecols=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
# print('{} description of {} users loaded'.format(len(users), max_userid))

# # movies = pd.read_csv(MOVIES_CSV, sep='::', engine='python',
# #         usecols=['movieID', 'Title', 'Genres'])
# movies = pd.read_csv('data/movies.csv',sep='::',engine='python')
# movies.columns = ['MovieID','Title','Type']
# print('{} descriptions of {} movies loaded'.format(len(movies), max_movieid))

# users['UserID'] = users.UserID.map(uid_encode).astype('int')
# movies['MovieID'] = movies.MovieID.map(mid_encode)
# # movies = movies.fillna(0)
# movies = movies.dropna()
# movies['MovieID'] = movies['MovieID'].astype('int')

# test_data = pd.read_csv(TEST_CSV, usecols=['UserID', 'MovieID'])
# test_data['UserID'] = test_data.UserID.map(uid_encode).astype('int')
# test_data['MovieID'] = test_data.MovieID.map(mid_encode).astype('int')
# print('{} testing data loaded.'.format(test_data.shape[0]))

# trained_model = get_model(uid_num,mid_num,bias,latent_dim)
# print('Loading model weights...')
# trained_model.load_weights('models/True_False_True_128_256.h5')
# print('Loading model done!!!')

# movies_array = movies.as_matrix()
# genres_map = {}
# for i in range(movies_array.shape[0]):
#     genre = movies_array[i][2].split('|')[0]
#     if genre not in genres_map.keys():
#         genres_map[genre] = [movies_array[i][0] - 1]
#     else:
#         genres_map[genre].append(movies_array[i][0] - 1)
# # print(genres_map)
# movie_emb = np.array(trained_model.layers[3].get_weights()).squeeze()
# model = TSNE(n_components=2, random_state=0)
# movie_emb = model.fit_transform(movie_emb)
# # print(key, genres_map[key])
# for key in genres_map.keys():
#     genres_map[key] = movie_emb[genres_map[key]]

# new_genres_map = {}
# for c in classes:
#     new_genres_map[c] = np.ndarray(shape=(0, 2))
#     for g in c.split('|'):
#         new_genres_map[c] = np.concatenate((new_genres_map[c], genres_map[g]), axis=0)
# #     print(new_genres_map[c].shape)
# draw(new_genres_map, 'graph.png')

