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



# load data 
map_dict = pk.load(open('data/map.pkl' , 'rb'))
X = np.load('data/X_test.npy')
name = pd.read_csv('data/sample_submission.csv')
X_name = name['fname'].tolist()



head = list(range(0,41))
head = [str(x) for x in head]


csv_folder = 'predict_csv_2'

if not os.path.exists(csv_folder):
    os.mkdir(csv_folder)

# # predict ================================================
# # write to csv


# # folders_1  = ['old_model/resnet_verified_refine_nonNormalize' ,
# #             'resnet_verified_refine_nonNormalize_semi', ]


folders_3  = ['model_full_resnet2_refine_co']

for f in folders_3:
     for path in listdir(f) :
         model = load_model(join(f,path))
         result = model.predict(X,verbose = 1 ,batch_size = 32)

         df = pd.DataFrame(result)
         df.insert(0,'fname' , X_name)
         df.to_csv(join(csv_folder , 'mike_resnet2__co_{}.csv'.format(path)), index=False)



# csv voting=========================================================================================================================

score = np.zeros((9400,41))


for path in listdir(csv_folder) :
        print(join(csv_folder , path))
        df = pd.read_csv(join(csv_folder,path))
        del df['fname']
        v = df.values
        score += v


# pick best 3 class
output = []
reverse_dict = pk.load(open('data/map_reverse.pkl' , 'rb'))

for i , d in enumerate( score):

    top3 = d.argsort()[-3:][::-1]
    result = [reverse_dict[x] for x in top3]
    s = ' '.join(result)

    output.append(s)
    
    
df = pd.DataFrame(output , columns = ['label'])
df.insert(0,'fname' , X_name)
df.to_csv('ans_resnet2.csv', index=False)
print(df.head(3))








