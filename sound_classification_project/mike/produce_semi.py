import numpy as np 
from sklearn.model_selection import KFold
import pickle as pk 
import os
from os import listdir
import pandas as pd
from os.path import isfile, join
from keras.utils import to_categorical ,Sequence
from sklearn.utils import shuffle

# create unverified data numpy ============================
'''
pd.options.mode.chained_assignment = None  # default='warn'
map_dict = pk.load(open('data/map.pkl' , 'rb'))
df = pd.read_csv('data/train_label.csv') 
audio_martix = np.load('data/mfcc/X_train.npy') 
df_unmanu = df[df['manually_verified'] == 0]
df_unmanu['trans'] = df_unmanu['label'].map(map_dict)
df_unmanu['onehot'] = df_unmanu['trans'].apply(lambda x: to_categorical(x,num_classes=41))
unmanu_veri_idx = df_unmanu.index.values
un_fnames = df_unmanu['fname'].values

X = audio_martix[unmanu_veri_idx]
Y = df_unmanu['onehot'].tolist()
Y = np.array(Y)
Y = Y.reshape(-1 ,41)

fold_path = 'data/semi_produce'
if not os.path.exists(fold_path):
    os.mkdir(fold_path)

np.save( os.path.join(fold_path, 'X_unverified'), X)
np.save( os.path.join(fold_path, 'Y_unverified'), Y)
np.save( os.path.join(fold_path, 'fname_unverified'), un_fnames)

print(df_unmanu.shape)
print(df_unmanu.head(3))
'''
# ==========================================================
fold_path = 'data/semi_produce'
X = np.load(os.path.join(fold_path, 'X_unverified'))
Y = np.load(os.path.join(fold_path, 'Y_unverified'))
fname_unverified = np.load(os.path.join(fold_path, 'fname_unverified'))

# create semi.py
mypath = 'old_model/resnet_verified_refine_nonNormalize'
models = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
print(models)


for i, m  in  enumerate( models ):
    print('#'*50)
    print('fold:{}'.format(i))

    model = load_model(m)
    result = model.predict(X,verbose = 1 ,batch_size = 256)

    break