import pandas as pd
import numpy as np
import time
import h5py
from keras.models import Model, load_model, Input
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization,PReLU, AveragePooling2D, GlobalAveragePooling2D, Average
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.activations import relu, softmax, softplus
from keras import losses, models, optimizers
import pickle as pk
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import KFold
import os
import shutil
#load data
map_dict = pk.load(open('data/map.pkl' , 'rb'))

#X = np.load('data/train/train3_X.npy')# 80% training data (7658, 40, 345,1)
X = np.load('data/train/X_train.npy') #100 training data
Y_train = pd.read_csv('data/train/train_label.csv')
Y_train['trans'] = Y_train['label'].map(map_dict)
Y_train['onehot'] = Y_train['trans'].apply(lambda x: to_categorical(x,num_classes=41))

Y = Y_train['onehot'].tolist()
print('X shape : ')
print(X.shape)

#print('Y_train shape : ')
#print(Y_train.head(10))

print('Y onehot shape :')
Y = np.array(Y)
Y = Y.reshape(-1 , 41)
print(Y.shape)

# Normalization
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean)/std

# set input shape and model_input
#input_shape = pixel_matrix[0,:,:,:].shape
#model_input = Input(shape=input_shape)



# Model:
def get_2d_conv_model(data):

    nclass = len(Y[0])
    
    # print(data.shape)
    inp = Input(shape=(data.shape))
    # print(inp)
    x = Conv2D(32, (4,10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)
    
    x = Conv2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)
    
    x = Conv2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)
    
    x = Conv2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(0.0005)
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model

# Model4:Original-CNN
def origin_cnn(data):
    nclass = len(Y[0])
    inp = Input(shape=data.shape)

    x = Conv2D(32, kernel_size=(4,10), activation='relu', padding='same')(inp)
    x = Dropout(0.25)(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, kernel_size=(4,10), activation='relu', padding='same')(x)
    x = Dropout(0.35)(x)
    x = AveragePooling2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, kernel_size=(4,10), activation='relu', padding='same')(x)                                                     
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, kernel_size=(4,10), activation='relu', padding='same')(x)                                                     
    x = AveragePooling2D()(x)
    x = Dropout(0.4)(x)
    
    x = Conv2D(256, kernel_size=(4,10), activation='relu', padding='same')(x)                                                     
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, kernel_size=(4,10), activation='relu', padding='same')(x)                                                     
    x = AveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    out = Dense(nclass, activation=softmax)(x)
    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(0.0005)
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model


PREDICTION_FOLDER = "predictions_origin_cnn"
target = '100%train'
if not os.path.exists('model/'+ PREDICTION_FOLDER):
    os.mkdir('model/' + PREDICTION_FOLDER)

if os.path.exists('logs/' + PREDICTION_FOLDER):
    shutil.rmtree('logs/' + PREDICTION_FOLDER)

kf = KFold(n_splits=6)

i = 0
for train_index, test_index in kf.split(X):
    i +=1 

    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # checkpoint = ModelCheckpoint('model/best_%d.h5'%i, monitor='val_loss', verbose=1, save_best_only=True)
    checkpoint = ModelCheckpoint('model/' + PREDICTION_FOLDER + '/'+ str(time.strftime("%m-%d"))+ '_'+ target +'_best_%d.h5'%i, monitor='val_loss', verbose=1, save_best_only=True)

    # early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

    tb = TensorBoard(log_dir='./logs/' + PREDICTION_FOLDER + '/' + str(time.strftime("%m-%d"))+ '_' + target + '_fold_%i'%i, write_graph=True)

    callbacks_list = [checkpoint, early, tb]

    print("#"*50)
    print("Fold: ", i)

    model = origin_cnn(X_train[0])

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), callbacks=callbacks_list, 
                        batch_size=32, epochs=100)

    
# Model1:ConvPool-CNN-C
def conv_pool_cnn(model_input):
    
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding =    'same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(7, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)
    
    model = Model(model_input, x, name='conv_pool_cnn')
    
    return model
# Model2: ALL-CNN-C
def all_cnn(model_input):
    
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding = 'same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same', strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same', strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(7, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)
        
    model = Model(model_input, x, name='all_cnn')
    
    return model

# Model3: Network in Network CNN
def nin_cnn(model_input):
    
    #mlpconv block 1
    x = Conv2D(32, (5, 5), activation='relu',padding='valid')(model_input)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    
    #mlpconv block2
    x = Conv2D(64, (3, 3), activation='relu',padding='valid')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    
    #mlpconv block3
    x = Conv2D(128, (3, 3), activation='relu',padding='valid')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(7, (1, 1))(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)
    
    model = Model(model_input, x, name='nin_cnn')
    
    return model
# Initiate the model and start Training
'''
conv_pool_cnn_model = conv_pool_cnn(model_input)
all_cnn_model = all_cnn(model_input)
nin_cnn_model = nin_cnn(model_input)
m1 = compile_and_train(conv_pool_cnn_model, num_epochs=20)
m2 = compile_and_train(all_cnn_model, num_epochs=20)
m3 = compile_and_train(nin_cnn_model, num_epochs=20)
origin_cnn_model = origin_cnn(model_input)
m4 = compile_and_train(origin_cnn_model, num_epochs=20)
'''

# Reinitiate the model
'''
conv_pool_cnn_model = conv_pool_cnn(model_input)
all_cnn_model = all_cnn(model_input)
nin_cnn_model = nin_cnn(model_input)
origin_cnn_model = origin_cnn(model_input)
'''

'''
#load the best saved weights
conv_pool_cnn_model.load_weights('output/weights/conv_pool_cnn.19-0.26.hdf5')
all_cnn_model.load_weights('output/weights/all_cnn.19-0.14.hdf5')
nin_cnn_model.load_weights('output/weights/nin_cnn.19-1.37.hdf5')
origin_cnn_model.load_weights('output/weights/origin_cnn.19-0.84.hdf5')
'''
#models = [conv_pool_cnn_model, all_cnn_model, nin_cnn_model, origin_cnn_model]

#Start ensemble-stacking
def ensemble(models, model_input):
    
    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)
    
    model = Model(model_input, y, name='ensemble')
    
    return model
'''
ensemble_model = ensemble(models, model_input)

conv_pool_cnn_model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc'])
all_cnn_model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc']) 
nin_cnn_model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc']) 
origin_cnn_model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc'])
ensemble_model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc']) 

scores = []
score1 = conv_pool_cnn_model.evaluate(pixel_valid_matrix, valid_y_oneHot)
score2 = all_cnn_model.evaluate(pixel_valid_matrix, valid_y_oneHot)
score3 = nin_cnn_model.evaluate(pixel_valid_matrix, valid_y_oneHot)
score5 = origin_cnn_model.evaluate(pixel_valid_matrix, valid_y_oneHot)
score4 = ensemble_model.evaluate(pixel_valid_matrix, valid_y_oneHot)
print(score1)
print(score2)
print(score3)
print(score5)
print(score4)
'''

