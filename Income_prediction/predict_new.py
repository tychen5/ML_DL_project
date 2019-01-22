
# coding: utf-8

# # Ensemble Prediction

# ## load clasifier

# In[3]:


import os, sys
import numpy as np
from random import shuffle
import argparse
from math import log, floor
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import *
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn.gaussian_process import *
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import *
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import *
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GridSearchCV
import time
import pickle
import tensorflow as tf
import tensorboard as tb
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.activations import *
from keras.callbacks import *
from keras.utils import *
from keras.layers.advanced_activations import *
from keras.layers.advanced_activations import *
from keras import *
from keras.engine.topology import *
from keras.optimizers import *
import keras


# In[4]:


import h5py


# In[5]:


with open('model/c_acc1_7.pickle', 'rb') as f:
    acc_list = pickle.load(f)
    
with open('model/neigh_c_2.pickle', 'rb') as f:
    neigh = pickle.load(f) #0

    
with open('model/svc_c_3.pickle', 'rb') as f:
    svc = pickle.load(f) #1

with open('model/dtc_c_1.5.pickle', 'rb') as f:
    dtc = pickle.load(f) #3

with open('model/mlp_c_2.pickle', 'rb') as f:
    mlp = pickle.load(f) #4

    
with open('model/LR_c_3.pickle', 'rb') as f:
    LR = pickle.load(f) #5


with open('model/lsvc_c_3.pickle', 'rb') as f:
    lsvc = pickle.load(f) #6
    
# with open('model/c_acc1_7.pickle', 'rb') as f:
    


# In[6]:


with open('model/r_acc1_7.pickle', 'rb') as f:
    acc_list_r = pickle.load(f)
    
with open('model/neigh_r.pickle', 'rb') as f:
    neigh_r = pickle.load(f) #1

    
with open('model/svr_r.pickle', 'rb') as f:
    svr = pickle.load(f) #2

with open('model/dtr_r.pickle', 'rb') as f:
    dtr = pickle.load(f) #4

with open('model/mlp_r.pickle', 'rb') as f:
    mlp_r = pickle.load(f) #5

    
with open('model/LR_r.pickle', 'rb') as f:
    LR_r = pickle.load(f) #X


with open('model/lsvr_r.pickle', 'rb') as f:
    lsvr = pickle.load(f) #7

with open('model/reg_r.pickle','rb') as f:
    reg = pickle.load(f) #8

with open('model/las_r.pickle','rb') as f:
    las = pickle.load(f) #9
    
with open('model/en_r.pickle','rb') as f:
    en = pickle.load(f) #10
    
with open('model/omg_r.pickle','rb') as f:
    omg = pickle.load(f) #11
    
with open('model/br_r.pickle','rb') as f:
    br = pickle.load(f) #12
    
with open('model/ardr_r.pickle','rb') as f:
    ardr = pickle.load(f) #13
    
with open('model/tsr_r.pickle','rb') as f:
    tsr = pickle.load(f) #14
    
# with open('model/c_acc1_7.pickle', 'rb') as f:
    


# In[7]:


def load_data(train_data_path, train_label_path, test_data_path,cols=None):
    if cols == None:
        X_train = pd.read_csv(train_data_path, sep=',', header=0)
        X_test = pd.read_csv(test_data_path, sep=',', header=0)
    else:
        X_train = pd.read_csv(train_data_path, sep=',')
        X_train['divide'] = X_train['capital_gain'].astype(float)/X_train['hours_per_week'].astype(float)
        X_train['minus'] = X_train['capital_gain']-X_train['capital_loss']
        X_train = X_train.filter(items=cols)
        
        X_test = pd.read_csv(test_data_path, sep=',')
        X_test['divide'] = X_test['capital_gain'].astype(float)/X_test['hours_per_week'].astype(float)
        X_test['minus'] = X_test['capital_gain']-X_test['capital_loss']
        X_test = X_test.filter(items=cols)
    X_train = np.array(X_train.values)
    Y_train = pd.read_csv(train_label_path, sep=',', header=0)
    Y_train = np.array(Y_train.values)
    X_test = np.array(X_test.values)
#     print(X_train.shape,Y_train.shape)
    return (X_train, Y_train, X_test)


# In[8]:


def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test


# In[9]:


##REG#
def NN_model2():    
    model = Sequential()
    model.add(Dense(6144, input_dim=77, kernel_initializer='uniform', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
#     model.add(Dropout(0.33))
    model.add(BatchNormalization())
    model.add(Dense(4096, kernel_initializer='uniform', activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
#     model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Dense(512, kernel_initializer='uniform', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
#     model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Dense(64, kernel_initializer='uniform', activation='relu', kernel_regularizer=regularizers.l2(0.00005)))
#     model.add(Dropout(0.2))
    model.add(BatchNormalization())
#     model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
#     model.add(Dropout(0.1))
#     model.add(BatchNormalization())
#     model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
#     model.add(Dropout(0.1))
#     model.add(BatchNormalization())

# model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

 ######################################   
    ##categorical
#     model.add(Dense(2,kernel_initializer='uniform')) #,activation='sigmoid
#     model.add(BatchNormalization())
#     model.add(Activation('softmax'))

#     model.add(Dense(2,activation='softmax'))

#     ADAM = Adam(amsgrad=True)
#     model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
 
 ###################################################   
    #regression:
    # model.add(PReLU(alpha_initializer='zero', weights=None))
    model.add(Dense(8,kernel_initializer='uniform'))
    # model.add(BatchNormalization())
    # model.add(PReLU(alpha_initializer='zero', weights=None))
    model.add(Activation('relu'))
#     model.add(Dropout(0.05))
    model.add(BatchNormalization())
    
#     model.add(Dense(1,kernel_initializer='uniform')) #,activation='sigmoid
#     model.add(BatchNormalization())
#     model.add(Activation('sigmoid'))

    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    opt = Adam(decay=1e-8) #amsgrad=False
#     opt = Adamax(lr=0.002,decay=1e-8)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy']) #binary_crossentropy
    
#################################################################################################################    
    model.summary()
    batch_size = 32#128 
    early_stopping_monitor = EarlyStopping(patience=20) #,monitor='val_acc',mode='auto' #15
    # keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    callback = GetBest(monitor='val_acc', verbose=1, mode='auto')
    # CW = {0:8.312,1:1.}
#     model.fit(X_T, y_T, batch_size=batch_size, epochs=500, validation_data=(x_valid,y_valid), callbacks=[callback,early_stopping_monitor],shuffle=True)#,class_weight=CW)
    
    return model


# In[10]:


##STACKING##
def NN_model3():
    model = Sequential()
    model.add(Dense(8192,activation='relu',input_dim=72, kernel_regularizer=regularizers.l2(0.00029))) #411,29
    model.add(Dropout(0.419))
    model.add(BatchNormalization())
    model.add(Dense(8192,activation='relu',kernel_regularizer=regularizers.l2(0.00029))) #411,29
    model.add(Dropout(0.419))
    model.add(BatchNormalization())
    model.add(Dense(128,activation='relu')) #128
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(128,activation='relu')) #128
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(2,activation='softmax'))

 ######################################   
    ##categorical
#     model.add(Dense(2,kernel_initializer='uniform')) #,activation='sigmoid
#     model.add(BatchNormalization())
#     model.add(Activation('softmax'))

#     model.add(Dense(2,activation='softmax'))
    opt = RMSprop(lr=0.00008, decay=1e-7)
#     opt = Nadam(lr=0.0002) #decay1e-8

#     ADAM = Adam(amsgrad=True)
#     model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
 
 ###################################################   

    
#################################################################################################################    
    model.summary()
    batch_size = 32#128 
    early_stopping_monitor = EarlyStopping(patience=20) #,monitor='val_acc',mode='auto' #15
    # keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    callback = GetBest(monitor='val_acc', verbose=1, mode='auto')
    # CW = {0:8.312,1:1.}
#     model.fit(X_T, y_T, batch_size=batch_size, epochs=500, validation_data=(x_valid,y_valid), callbacks=[callback,early_stopping_monitor],shuffle=True)#,class_weight=CW)
    
    return model


# In[11]:


import numpy as np
from keras.callbacks import Callback
class GetBest(Callback):
    """Get the best model at the end of training.
	# Arguments
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        mode: one of {auto, min, max}.
            The decision
            to overwrite the current stored weights is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
	# Example
		callbacks = [GetBest(monitor='val_acc', verbose=1, mode='max')]
		mode.fit(X, y, validation_data=(X_eval, Y_eval),
                 callbacks=callbacks)
    """

    def __init__(self, monitor='val_loss', verbose=0,
                 mode='auto', period=1):
        super(GetBest, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.best_epochs = 0
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('GetBest mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
                
    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            #filepath = self.filepath.format(epoch=epoch + 1, **logs)
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can pick best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' storing weights.'
                              % (epoch + 1, self.monitor, self.best,
                                 current))
                    self.best = current
                    self.best_epochs = epoch + 1
                    self.best_weights = self.model.get_weights()
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve' %
                              (epoch + 1, self.monitor))            
                    
    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print('Using epoch %05d with %s: %0.5f' % (self.best_epochs, self.monitor,
                                                       self.best))
        self.model.set_weights(self.best_weights)


# In[17]:


def predict1(X_test, save_dir):
    test_data_size = len(X_test)

    # Load parameters
    print('=====Loading Param from %s=====' % save_dir)
    model = NN_model2()
    model.load_weights(os.path.join(save_dir, 'reg_NN_weights.h5'))  #model 名稱


    # predict
    ans = model.predict(X_test)
    
    
    #regression  #ensemble
#     anss=[]
#     for aa in ans:
#         if aa[0] < 0.5:
#             anss.append(0)
#         else:
#             anss.append(1)
#     print(np.mean(anss))
    
    #categorical:
#     anss = argmax(to_categorical(ans,2),axis=1)


#     print('=====Write output to %s =====' % output_dir)
#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)
#     output_path = os.path.join(output_dir, 'reg_DNN_08X2_aug.csv')  ##輸出檔案名稱
#     with open(output_path, 'w') as f:
#         f.write('id,label\n')
#         for i, v in  enumerate(anss):
#             f.write('%d,%d\n' %(i+1, v))
# #             f.write('%d,%d\n' %(i+1, v[1]))  #categorical


    return ans


# In[19]:


def predict2(X_test, save_dir):
    test_data_size = len(X_test)

    # Load parameters
    print('=====Loading Param from %s=====' % save_dir)
    model = NN_model3()
    model.load_weights(os.path.join(save_dir, 'clf_NN_weights.h5'))  #model 名稱


    # predict
    anss = model.predict(X_test)
    
    
    #regression  #ensemble
#     anss=[]
#     for aa in ans:
#         if aa[0] < 0.5:
#             anss.append(0)
#         else:
#             anss.append(1)
#     print(np.mean(anss))
    
    #categorical:
#     anss = argmax(to_categorical(ans,2),axis=1)
    ans = []
    for v in anss:
        ans.append(v[1])

    ans = np.asarray(ans)
#     print('=====Write output to %s =====' % output_dir)
#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)
#     output_path = os.path.join(output_dir, 'reg_DNN_08X2_aug.csv')  ##輸出檔案名稱
#     with open(output_path, 'w') as f:
#         f.write('id,label\n')
#         for i, v in  enumerate(anss):
#             f.write('%d,%d\n' %(i+1, v))
# #             f.write('%d,%d\n' %(i+1, v[1]))  #categorical


    return ans


# In[14]:


with open('model/clf_col.pickle','rb') as f:
    clf_col = pickle.load(f)
# len(clf_col)
with open('model/reg_col.picke', 'rb') as f:
    reg_col = pickle.load(f)
# len(reg_col)
with open('model/clf_colo.pickle','rb') as f:
    clf_colo = pickle.load(f)
# len(clf_col)
with open('model/reg_colo.picke', 'rb') as f:
    reg_colo = pickle.load(f)
# len(reg_col)


# In[15]:


X_all_c, Y_all, X_test_c = load_data('data/train_X', 'data/train_Y', 'data/test_X',clf_col)
X_all_r, Y_all, X_test_r = load_data('data/train_X', 'data/train_Y', 'data/test_X',reg_col)
X_all_co, Y_all, X_test_co = load_data('data/train_X', 'data/train_Y', 'data/test_X',clf_colo)
X_all_ro, Y_all, X_test_ro = load_data('data/train_X', 'data/train_Y', 'data/test_X',reg_colo)

X_all_c, X_test_c = normalize(X_all_c, X_test_c)
X_all_r, X_test_r = normalize(X_all_r, X_test_r)
X_all_co, X_test_co = normalize(X_all_co, X_test_co)
X_all_ro, X_test_ro = normalize(X_all_ro, X_test_ro)


# In[20]:



start_time = time.time()
y1 = neigh.predict(X_test_c)
y1_r = neigh_r.predict(X_test_r)
print("--- %s seconds ---" % (time.time() - start_time))
y2 = svc.predict(X_test_c)
y2_r = svr.predict(X_test_r)
print("--- %s seconds ---" % (time.time() - start_time))
y4 = dtc.predict(X_test_c)
y4_r = dtr.predict(X_test_r)

y5= mlp.predict(X_test_c)
y5_r =mlp_r.predict(X_test_r)

y6 = LR.predict(X_test_c)
y6_r = LR_r.predict(X_test_r)

y7 = lsvc.predict(X_test_c)
y7_r = lsvr.predict(X_test_r)

y8_r = reg.predict(X_test_r)
y9_r = las.predict(X_test_r)
y10_r = en.predict(X_test_r)
y11_r = omg.predict(X_test_r)
y12_r = br.predict(X_test_r)
y13_r = ardr.predict(X_test_r)
y14_r = tsr.predict(X_test_r)
print("--- %s seconds ---" % (time.time() - start_time))


##NN
y_DNN = predict1(X_test_ro, 'model/')
y_DNN = y_DNN.ravel()
y_DNN_c = predict2(X_test_co,'model/')
print("--- %s seconds ---" % (time.time() - start_time))


# In[13]:


# y_DNN_cc = np.asarray(y_DNN_c)
# y_DNN_cc *2 -1


# In[31]:


## 轉至0~1之間在變成-1~1

y_DNN[y_DNN<0]=0
y_DNN[y_DNN>1]=1
y1_r[y1_r<0]=0
y1_r[y1_r>1]=1
y2_r[y2_r<0]=0
y2_r[y2_r>1]=1
y4_r[y4_r<0]=0
y4_r[y4_r>1]=1
y5_r[y5_r<0]=0
y5_r[y5_r>1]=1
y6_r[y6_r<0]=0
y6_r[y6_r>1]=1
y7_r[y7_r<0]=0
y7_r[y7_r>1]=1
y8_r[y8_r<0]=0
y8_r[y8_r>1]=1
y9_r[y9_r<0]=0
y9_r[y9_r>1]=1
y10_r[y10_r<0]=0
y10_r[y10_r>1]=1
y11_r[y11_r<0]=0
y11_r[y11_r>1]=1
y12_r[y12_r<0]=0
y12_r[y12_r>1]=1
y13_r[y13_r<0]=0
y13_r[y13_r>1]=1
y14_r[y14_r<0]=0
y14_r[y14_r>1]=1
y_DNN = y_DNN*2-1
y_DNN_c = y_DNN_c*2-1
y1 = y1*2-1
y1_r = y1_r*2-1
y2 = y2*2-1
y2_r = y2_r *2-1
# y3 = y3*2-1
y4 = y4*2-1
y4_r = y4_r*2-1
y5 = y5*2-1
y5_r = y5_r*2-1
y6 = y6*2-1
y6_r = y6_r*2-1
y7 = y7*2-1
y7_r = y7_r*2-1
y8_r = y8_r*2-1
y9_r = y9_r*2-1
y10_r = y10_r*2-1
y11_r = y11_r*2-1
y12_r = y12_r*2-1
y13_r = y13_r*2-1
y14_r = y14_r*2-1


# In[32]:


print(acc_list)
print(acc_list_r)


# In[28]:


# y_DNN


# In[34]:



acc_DNN = 2.001
acc_DNN_c =1.978
acc1 = acc_list[0] #KN
acc1_r = acc_list_r[0]
acc2 = acc_list[1] #SV
acc2_r = acc_list_r[1]
# acc3 = acc_list[2] #GP
acc4 = acc_list[3] #dt
acc4_r = acc_list_r[3]
acc5 = acc_list[4] #mlp
acc5_r = acc_list_r[4]
acc6 = acc_list[5] #LR
acc6_r = acc_list_r[5]
# acc6_r = 0.8 #ad-hoc
acc7 = acc_list[6] #lsv
acc7_r = acc_list_r[6]
acc8_r = acc_list_r[7]
acc9_r = acc_list_r[8]
acc10_r = acc_list_r[9]
acc11_r = acc_list_r[10]
acc12_r = acc_list_r[11]
acc13_r = acc_list_r[12]
acc14_r = acc_list_r[13]
# ens = (y1*acc1*2+y2*acc2*3+y4*acc4*1.5+y5*acc5*2+y6*acc6*3+y7*acc7*3)/(acc1*2+acc2*3+acc4*1.5+acc5*2+acc6*3+acc7*3) #家群平均
ens = (y1*acc1+y2*acc2+y4*acc4+y5*acc5+y6*acc6+y7*acc7 + 
       y1_r*acc1_r+y2_r*acc2_r+y4_r*acc4_r+y5_r*acc5_r+y6_r*acc6_r+y7_r*acc7_r+y8_r*acc8_r+y9_r*acc9_r+y10_r*acc10_r+y11_r*acc11_r+y12_r*acc12_r+y13_r*acc13_r+y14_r*acc14_r 
       + y_DNN*acc_DNN+y_DNN_c*acc_DNN_c)/(acc1+acc2+acc4+acc5+acc6+acc7+
                                           acc1_r+acc2_r+acc4_r+acc5_r+acc6_r+acc7_r+acc8_r+acc9_r+acc10_r+acc11_r+acc12_r+acc13_r+acc14_r+
                                           acc_DNN+acc_DNN_c)
final_ans =[]
for v in ens: #轉回label
    if v < 0:
        final_ans.append(0)
    else:
        final_ans.append(1)
np.mean(final_ans)


# In[35]:


test_data_size = len(X_test_r)

# # Load parameters
# print('=====Loading Param from %s=====' % save_dir)
# w = np.loadtxt(os.path.join(save_dir, 'w'))
# b = np.loadtxt(os.path.join(save_dir, 'b'))

# # predict
# z = (np.dot(X_test, np.transpose(w)) + b)
# y = sigmoid(z)
# y_ = np.around(y)
output_dir = 'result/'
print('=====Write output to %s =====' % output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_path = os.path.join(output_dir, 'test.csv')
with open(output_path, 'w') as f:
    f.write('id,label\n')
    for i, v in  enumerate(final_ans):
        f.write('%d,%d\n' %(i+1, v))

