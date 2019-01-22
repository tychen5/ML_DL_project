
# coding: utf-8

# In[31]:


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
    from keras.regularizers import l2
    from keras.initializers import Zeros
    from keras.callbacks import History ,ModelCheckpoint, EarlyStopping
except:
    pass


# In[32]:


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


# In[33]:


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


# In[34]:


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


# In[35]:


test_path = sys.argv[1]
predict_path = sys.argv[2]#'result/res.csv'
movie_path = sys.argv[3]
user_path = sys.argv[4]

movies = pd.read_csv(movie_path,sep='::',engine='python') # mid,title,type
movies.columns = ['MovieID','Title','Type']
movies['Year'] = movies['Title'].apply(lambda x:str(x)[-5:-1])
users = pd.read_csv(user_path,sep='::',engine='python') # gender, age,occupationmzip
users.columns = ['UserID','Gender','Age','Occupation','Zip']
# train = pd.read_csv('data/train.csv')
# test = pd.read_csv('data/test.csv')
test = pd.read_csv(test_path) 


uid_encode = pickle.load(open('models/uid_encode.pkl','rb'))
mid_encode = pickle.load(open('models/mid_encode.pkl','rb'))
age_encode = pickle.load(open('models/age_encode.pkl','rb'))
zip_encode = pickle.load(open('models/zip_encode.pkl','rb'))
year_encode = pickle.load(open('models/year_encode.pkl','rb'))
uid_num = len(uid_encode)
mid_num = len(mid_encode)

dp=0.69
lmbda = 3e-5
latent_dim = 64
fb_dim = 3


# In[36]:


test_df = test.filter(['UserID','MovieID'])
test_df = pd.merge(test_df,users , on='UserID',how='left')
test_df = pd.merge(test_df,movies, on='MovieID',how='left')
test_df = test_df.drop(['Title'],axis=1)
#test_df


# In[37]:


test_df['UserID'] = test_df.UserID.map(uid_encode)
test_df['MovieID'] = test_df.MovieID.map(mid_encode)
test_df['Age'] = test_df.Age.map(age_encode)
test_df['Zip'] = test_df.Zip.map(zip_encode)
test_df['Year'] = test_df.Year.map(year_encode)

test_df['Gender'] = test_df.Gender.map(gen2onehot)
test_df['Occupation'] = test_df.Occupation.map(occu2onehot)
test_df['Type'] = test_df.Type.map(type2onehot)
#test_df


# In[38]:


test_df_user = test_df.filter(['UserID'])

test_df_user_fb = test_df.filter(['Age','Zip'])
temp = pd.DataFrame(test_df['Gender'].values.tolist())
test_df_user_fb = pd.merge(temp,test_df_user_fb,how='right',right_index=True,left_index=True)
temp = pd.DataFrame(test_df['Occupation'].values.tolist())
test_df_user_fb = pd.merge(test_df_user_fb,temp,how='left',right_index=True,left_index=True)

test_df_movie = test_df.filter(['MovieID'])

test_df_movie_fb = test_df.filter(['Year'])
temp = pd.DataFrame(test_df['Type'].values.tolist())
test_df_movie_fb = pd.merge(test_df_movie_fb,temp,how='left',right_index=True,left_index=True)


# In[39]:


test_user = np.array(test_df_user)
test_user_fb = np.array(test_df_user_fb)
test_movie = np.array(test_df_movie)
test_movie_fb = np.array(test_df_movie_fb)
# trai_rating = np.array(train["Rating"])
print(test_user.shape , test_user_fb.shape, test_movie.shape, test_movie_fb.shape )


# In[40]:


rating_mean = 3.5817120860388076
rating_std = 1.116897661146206
def rmse(y_true, y_pred):
    y_true = y_true*rating_std+rating_mean
    y_pred = y_pred*rating_std+rating_mean
    y_pred = K.clip(y_pred, 1.0, 5.0) #rating range
    return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))


# ## Model

# In[41]:


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


# In[42]:


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


# In[43]:


model = get_model(uid_num,mid_num,dp=dp,lmbda=lmbda,latent_dim=latent_dim,fb_dim=fb_dim)
model.load_weights('models/0.7_2e-05_64_3_256.h5')
model2 = get_model2(uid_num,mid_num,dp=0.1,lmbda=1e-5,latent_dim=64)
model2.load_weights('models/get2_0.1_1e-05_64_3_2048.h5')


# In[44]:


ans = model.predict([test_user,test_user_fb,test_movie,test_movie_fb])
ans2 = model2.predict([test_user,test_user_fb,test_movie,test_movie_fb])
ans = ans*rating_std+rating_mean
ans2 = ans2*rating_std+rating_mean
ans = np.clip(ans,1,5)
ans2 = np.clip(ans2,1,5)
ans = (ans*2+ans2)/3
ans = np.clip(ans,1,5)


# In[45]:


# predict_path = 'result/reproduce55.csv'#'result/res_NN_MF.csv'
df = pd.DataFrame(ans,columns=['Rating'])
df['TestDataID']=df.index +1
df = df[['TestDataID','Rating']]
df.to_csv(predict_path,index=False)
print(predict_path)

