from keras.models import load_model
import numpy as np
import sys
import os
import pickle as pk
import pandas as pd


from keras.utils import to_categorical ,Sequence
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)

from keras.layers import (Convolution1D, Dense, Dropout, GlobalAveragePooling1D, 
                          GlobalMaxPool1D, Input, MaxPool1D, concatenate)

from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten,
                          GlobalMaxPool2D, MaxPool2D, concatenate, Activation)

from keras import backend as K       

# calculate accuracy
from sklearn.metrics import accuracy_score


model_num = 6

reverse_dict = pk.load(open('data/map_reverse.pkl' , 'rb'))
name = pd.read_csv('data/sample_submission.csv')
X_name = name['fname'].tolist()

# 1 val_loss = 1.6938
# 2 val_loss = 1.7576
# 3 val_loss = 1.6869
# 4 val_loss = 1.6283
# 5 val_loss = 1.7429
# 6 val_loss = 1.6532

output = []
X = np.load("data/X_test.npy")

# #  Normalization =====================================================
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean)/std


model = load_model('model_new/mike_six_fold_{}.h5'.format(model_num)) 
result = model.predict(X)
result = np.argmax(result, axis=-1)
ans = [ reverse_dict[x] for x in result ]


df = pd.DataFrame(ans , columns = ['label'])
df.insert(0,'fname' , X_name)
df.to_csv('ans_{}.csv'.format(model_num), index=False)



