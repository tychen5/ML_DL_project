import numpy as np 
from sklearn.model_selection import KFold
import pickle as pk 
import os
import pandas as pd

from keras.utils import to_categorical ,Sequence
from sklearn.utils import shuffle

pd.options.mode.chained_assignment = None  # default='warn'

map_dict = pk.load(open('data/map.pkl' , 'rb'))

df = pd.read_csv('data/train_label.csv')  

# audio_martix = np.load('data/raw/X_train.npy')
audio_martix = np.load('data/mfcc/X_train.npy')
test_X = np.load('data/X_test.npy')
# df_manu ================================================
df_manu = df[df['manually_verified'] == 1]
df_manu['trans'] = df_manu['label'].map(map_dict)
df_manu['onehot'] = df_manu['trans'].apply(lambda x: to_categorical(x,num_classes=41))

# manu index index 
manu_veri_idx = df_manu.index.values
fnames = df_manu['fname'].values

mean = np.mean(np.append(audio_martix,test_X , axis=0), axis=0)
std = np.std(np.append(audio_martix,test_X , axis=0), axis=0)
np.save('data/mean_std.npy', [mean , std])


# manu_train
X =  audio_martix[manu_veri_idx]
Y =  df_manu['onehot'].tolist()
Y = np.array(Y)
Y = Y.reshape(-1 ,41)

X , Y = shuffle(X, Y, random_state=5)

print(manu_veri_idx[0:10])
print(df_manu.head(3))
print(X.shape)
print(Y.shape)

fold_path = 'data/ten_fold_data'
if not os.path.exists(fold_path):
    os.mkdir(fold_path)


kf = KFold(n_splits=10)
k = 0
# split manu data
for train_index, test_index in kf.split(manu_veri_idx):
    k+=1
    X_train , X_valid = X[train_index], X[test_index]
    Y_train, Y_valid = Y[train_index], Y[test_index]
    Y_valid_fname = fnames[test_index]
    
    np.save( os.path.join(fold_path, 'X_train_{}'.format(k)), X_train)
    np.save( os.path.join(fold_path, 'Y_train_{}'.format(k)), Y_train)
    np.save( os.path.join(fold_path, 'X_valid_{}'.format(k)), X_valid)
    np.save( os.path.join(fold_path, 'Y_valid_{}'.format(k)), Y_valid)
    np.save( os.path.join(fold_path, 'valid_fname_{}'.format(k)), Y_valid_fname)
    print('{} fold split done'.format(k))

print('verified  data split done =====================')

# df_unmanu ================================================

df_unmanu = df[df['manually_verified'] == 0]
df_unmanu['trans'] = df_unmanu['label'].map(map_dict)
df_unmanu['onehot'] = df_unmanu['trans'].apply(lambda x: to_categorical(x,num_classes=41))
unmanu_veri_idx = df_unmanu.index.values

un_fnames = df_unmanu['fname'].values

X = audio_martix[unmanu_veri_idx]
Y = df_unmanu['onehot'].tolist()
Y = np.array(Y)
Y = Y.reshape(-1 ,41)

# mean = np.mean(X, axis=0)
# std = np.std(X, axis=0)
# X = (X - mean)/std

np.save( os.path.join(fold_path, 'X_unverified'), X)
np.save( os.path.join(fold_path, 'Y_unverified'), Y)
np.save( os.path.join(fold_path, 'fname_unverified'), un_fnames)


print(unmanu_veri_idx[0:10])
print(df_unmanu.head(3))
print(X.shape)
print(Y.shape)
print('unverified data done =====================')
