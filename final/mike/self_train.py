import os
import shutil
import numpy as np
import pickle as pk
import pandas as pd
from keras.utils import to_categorical ,Sequence
from keras import losses, models, optimizers
from keras.models import load_model
from keras.models import Sequential
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.layers import Activation, LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from sklearn.model_selection import KFold
from random_eraser import get_random_eraser
from keras.optimizers import Adam
from os.path import join
import resnet
from sklearn.utils import shuffle

map_dict = pk.load(open('data/map.pkl' , 'rb'))

semi = pd.read_csv('data/cotrain/Y_selftrain_ens_verified.csv')

semi_map = {}

semi_name = semi['fname'].values
semi_label_verified = semi['label_verified'].values

for idx ,d in enumerate( semi_name):
    semi_map[d] = semi_label_verified[idx]

unverified_df = pd.read_csv('data/train_label.csv')
test_df = pd.read_csv('data/sample_submission.csv')

unverified_df = unverified_df[unverified_df['fname'].isin(semi_name)]
unverified_df = unverified_df.drop(columns=['manually_verified'])
unverified_df['label_verified'] = unverified_df['fname'].map(semi_map)

test_df = test_df[test_df['fname'].isin(semi_name)]
test_df['label_verified'] = test_df['fname'].map(semi_map)

unverified_idx = unverified_df.index.values
test_idx = test_df.index.values

df = pd.concat([unverified_df , test_df])
df = df.drop(columns=['label'])
df['trans'] = df['label_verified'].map(map_dict)
df['onehot'] = df['trans'].apply(lambda x: to_categorical(x,num_classes=41))



X_unverified = np.load('data/mfcc/X_train.npy')[unverified_idx]
X_test = np.load('data/X_test.npy')[test_idx]

X_semi = np.append(X_unverified,X_test , axis=0)
Y_semi = np.array(df['onehot'].tolist()).reshape(-1,41)

print(X_semi.shape)
print(Y_semi.shape)

 # data generator ====================================================================================
class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y


model_path = 'model_full_resnet18_gen_refine'
refine_path = 'model_full_resnet18_gen_refine_co'

all_x = np.concatenate( (np.load('data/mfcc/X_train.npy') , np.load('data/X_test.npy')))

if not os.path.exists(refine_path):
    os.mkdir(refine_path)

for i in range(1,11):
    X_train = np.load('data/ten_fold_data/X_train_{}.npy'.format(i)) 
    Y_train = np.load('data/ten_fold_data/Y_train_{}.npy'.format(i)) 
    X_test = np.load('data/ten_fold_data/X_valid_{}.npy'.format(i))
    Y_test = np.load('data/ten_fold_data/Y_valid_{}.npy'.format(i))
    
    X_train = np.append(X_train,X_semi , axis=0)
    Y_train = np.append(Y_train,Y_semi , axis=0)
    
    X_train , Y_train = shuffle(X_train, Y_train, random_state=5)
    
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
    
    model = load_model(join(model_path,'semi_{}.h5'.format(i)))
    
    checkpoint = ModelCheckpoint(join(refine_path , 'semi_self_%d_{val_acc:.5f}.h5'%i), monitor='val_acc', verbose=1, save_best_only=True)
    early = EarlyStopping(monitor="val_acc", mode="max", patience=30)
    callbacks_list = [checkpoint, early]
    
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        preprocessing_function=get_random_eraser(v_l=np.min(all_x), v_h=np.max(all_x)) # Trainset's boundaries.
    )
    
    mygenerator = MixupGenerator(X_train, Y_train, alpha=1.0, batch_size=128, datagen=datagen)
    
    model.compile(loss='categorical_crossentropy',
             optimizer=Adam(lr=0.0001),
             metrics=['accuracy'])
    # mixup
    history = model.fit_generator(mygenerator(),
                    steps_per_epoch= X_train.shape[0] // 128,
                    epochs=10000,
                    validation_data=(X_test,Y_test), callbacks=callbacks_list)
    # normalize
#     history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), callbacks=callbacks_list,
#                         batch_size=32, epochs=10000)

    
#     break
