from keras.models import load_model
import numpy as np
import sys
import os
import pickle as pk
import pandas as pd
from os import listdir
from os.path import isfile, join


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

# processing ==================================
# predict_acc.py
# final_csv_voting
# output file 
# =============================================

map_dict = pk.load(open('data/map.pkl' , 'rb'))
reverse_dict = pk.load(open('data/map_reverse.pkl' , 'rb'))
name = pd.read_csv('data/sample_submission.csv')
X_name = name['fname'].tolist()


folders  = ['resnet_semi_test_csv' , 'cnn2d_semi_test_csv']


result = np.zeros((9400,41))

for f in folders:
    for path in listdir(f) :
        
        df = pd.read_csv(join(f,path))
        del df['fname']
        v = df.values
        result += v
        


output = []
reverse_dict = pk.load(open('data/map_reverse.pkl' , 'rb'))

for i , d in enumerate( result):

    top3 = d.argsort()[-3:][::-1]
    result = [reverse_dict[x] for x in top3]
    s = ' '.join(result)

    output.append(s)

df = pd.DataFrame(output , columns = ['label'])
df.insert(0,'fname' , X_name)
df.to_csv('ans_csv_voting.csv', index=False)
print(df.head(3))