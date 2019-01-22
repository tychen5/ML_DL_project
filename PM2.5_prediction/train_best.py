
# coding: utf-8

# In[1]:


import csv
import sys
import numpy as np
import pandas as pd
import sklearn
from sklearn.svm import SVR
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import *
from sklearn.model_selection import GridSearchCV   ##Grid Search CV(model,parameter_dict) ; 
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import pickle
from math import sqrt
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.activations import *
from keras.callbacks import *
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.utils import *
from keras.layers.advanced_activations import *
from keras.layers.advanced_activations import *
from keras import *
from keras.engine.topology import *
from keras.optimizers import *
# import re


# *** Train Data ***

# In[2]:


data = []
f = open('./data/train.csv', 'r', encoding = 'big5')
train_title = f.readline()

for line in csv.reader(f):
    
    # get data from column index 3 and after
    # replace NR by 0 
    the_data = [float(i.replace('NR','-1')) for i in line[3:]]
    
    data.append(the_data)
    
f.close()
X = []
y = []


# In[3]:


for month in range(1,13):
    
    # 每月資料有 18 feature * 20 day =  360 筆 data
    month_data = data[360 * (month-1):360 * month]
    
    
    # 建立 feature_list[0] ~ feature_list[18]
    # feature_list[i] 有  24 hr * 20 day = 480 筆資料
    
    feature_id = list(range(18))
    
    feature_list = []
    
    for fid in range(18):
        feature_list.append([])
        
    for i, row in enumerate(month_data):
        the_fid = i % 18
        for stat in row:
            feature_list[the_fid].append(stat)
            
            
    # PM2.5 在 index 9
    # 因為是用前 9 小時資料來預測 PM2.5
    # 因此每個月前 9 筆  PM2.5 不能作為 training data
            
    for j, label in enumerate(feature_list[9][9:], 9):
        take_to_train = True
        stat_list = []
        # loop 18 features
        for fid in feature_id:
            # loop prev 9 hr
            for feature in feature_list[fid][j-9:j]:
                                
                stat_list.append(feature)
                
        stat_list = np.asarray(stat_list)
        stat_list[stat_list < 0] = 0
        stat_list = stat_list.tolist()
                
        

        
        if(take_to_train):
                        
            X.append(stat_list)
            y.append(label)
        
        
X_T = np.asarray(X)
y_T = np.asarray(y)


# In[4]:


train_valid_ratio = 0.99
indices = np.random.permutation(X_T.shape[0])
train_idx, valid_idx = indices[:int(X_T.shape[0] * train_valid_ratio)], indices[int(X_T.shape[0] * train_valid_ratio):]
X_T, x_valid = X_T[train_idx,:], X_T[valid_idx,:]
y_T, y_valid = y_T[train_idx], y_T[valid_idx]


# In[7]:


X_T = X_T[y_T >= 0]
x_valid = x_valid[y_V>=0]
y_T = y_T[y_T >= 0]
y_valid = y_valid[y_V>=0]
model_dict={}

X_V = x_valid
y_V = y_valid

# ***

# In[ ]:


regr_2 = AdaBoostRegressor(DecisionTreeRegressor(),
                          n_estimators=200)   ## max_depth=15, n_estimator=200=>valid:7.2
regr_2.fit(X_T, y_T)
model_dict['ada']=regr_2
y_a = regr_2.predict(X_V)
# np.sum(y_V - y_2)
rmse = sqrt(mean_squared_error(y_V,y_a))
print(rmse)


# In[ ]:


xgbb =xgb.XGBRegressor(eval_metric='rmse',booster='gbtree',learning_rate=0.1,min_child_weight=2,max_depth=2,subsample=0.93
                       ,colsample_bytree=0.78,gamma=0,n_jobs=-1) #n_jobs=-1) #,max_depth=2 #subsample=0.93 #,colsample_bytree=0.78 #,gamma=0
xgbb.fit(X_T,y_T)
model_dict['xgb']=xgbb
y_2 = xgbb.predict(X_V)
# y_2 = clf.predict(X_V)
rmse = sqrt(mean_squared_error(y_V,y_2))
print(rmse)


# In[ ]:


rdf = RandomForestRegressor(criterion='mse',bootstrap=True,max_features='auto'
                            ,min_samples_leaf=3,min_samples_split=13,n_estimators=250,n_jobs=-1) #,max_depth=8
rdf.fit(X_T, y_T)
model_dict['rdf']=rdf
y_2 = rdf.predict(X_V)
# y_2 = clf.predict(X_V)
rmse = sqrt(mean_squared_error(y_V,y_2))
print(rmse)


# In[ ]:





svmr = SVR(kernel='linear')
scorer = make_scorer(mean_squared_error, greater_is_better=False)
svmr.fit(X_T,y_T)
model_dict['svm']=svmr

y_2 = svmr.predict(X_V)
# y_2 = clf.predict(X_V)
rmse = sqrt(mean_squared_error(y_V,y_2))
print(rmse)



# In[75]:


rmse


# ### 以下DL請注意

# In[74]:


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


# In[10]:


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


# In[26]:


model =Sequential()
# model.add(LSTM(1024,input_shape=(9,12),dropout=0.25)) #,return_sequences=True))會要每個小時都會過一次DENSE需要對答案
model.add(Dense(512,kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.00001),input_dim=162))
model.add(BatchNormalization())
# model.add(PReLU(alpha_initializer='zero', weights=None))
model.add(Activation('selu'))
model.add(Dropout(0.2))
model.add(Dense(512,kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.00001))) #,input_dim = 442
model.add(BatchNormalization())
# model.add(PReLU(alpha_initializer='zero', weights=None))
model.add(Activation('selu'))
model.add(Dropout(0.1))
model.add(Dense(128,kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.00001)))
model.add(BatchNormalization())
# model.add(PReLU(alpha_initializer='zero', weights=None))
model.add(Activation('selu'))
model.add(Dropout(0.1))
model.add(Dense(128,kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.00001)))
model.add(BatchNormalization())
# model.add(PReLU(alpha_initializer='zero', weights=None))
model.add(Activation('selu'))
model.add(Dropout(0.05))
model.add(Dense(32,kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.00001)))
# model.add(BatchNormalization())
# model.add(PReLU(alpha_initializer='zero', weights=None))
model.add(Activation('relu'))
model.add(Dropout(0.05))
model.add(BatchNormalization())
model.add(Dense(32,kernel_initializer='uniform'))
# model.add(BatchNormalization())
# model.add(PReLU(alpha_initializer='zero', weights=None))
model.add(Activation('relu'))
model.add(BatchNormalization())
# model.add(BatchNormalization())
# model.add(PReLU(alpha_initializer='zero', weights=None))
model.add(Dense(8,kernel_initializer='uniform'))
# model.add(BatchNormalization())
# model.add(PReLU(alpha_initializer='zero', weights=None))
model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(BatchNormalization())

# model.add(PReLU(alpha_initializer='zero', weights=None))
model.add(Dense(8,kernel_initializer='uniform'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(PReLU(alpha_initializer='zero', weights=None))
model.add(BatchNormalization())
# model.add(Dense(512,input_dim=162,kernel_initializer='uniform'))
# model.add(BatchNormalization())
# # model.add(PReLU(alpha_initializer='zero', weights=None))
# model.add(Activation('relu'))
# model.add(Dense(512,kernel_initializer='uniform'))
# model.add(BatchNormalization())
# # model.add(PReLU(alpha_initializer='zero', weights=None))
# model.add(Activation('relu'))
# model.add(Dense(256,kernel_initializer='uniform'))
# model.add(BatchNormalization())
# # model.add(PReLU(alpha_initializer='zero', weights=None))
# model.add(Activation('relu'))
# model.add(Dense(256,kernel_initializer='uniform'))
# model.add(BatchNormalization())
# model.add(PReLU(alpha_initializer='zero', weights=None))
# model.add(Activation('relu'))
# model.add(Dense(32))
# model.add(Dense(32))
# model.add(Dense(32))

model.add(Dense(1))
model.compile(loss=root_mean_squared_error, optimizer='rmsprop', metrics=[root_mean_squared_error]) #binary_crossentropy
model.summary()


# In[27]:


batch_size = 16#128
early_stopping_monitor = EarlyStopping(patience=10)
callback = GetBest(monitor='val_root_mean_squared_error', verbose=1, mode='min')
# X_T = X_T.reshape(-1,18)
# x_valid = x_valid.reshape(-1,18)
model.fit(X_T, y_T, batch_size=batch_size, epochs=200, validation_data=(X_V,y_V), callbacks=[early_stopping_monitor,callback],shuffle=True) #重作earlystop

##hooklog####


# In[44]:


# model_dict['dnn']=model
# x_v = X_V.reshape(-1,18)
ans = model.predict(X_V)
# ans * norm_dict
# ans = reverse_normalize(ans)
# y_V = reverse_normalize(y_V)
rmse = sqrt(mean_squared_error(y_V,ans))
# rmse = sqrt(mean_squared_error(reverse_normalize(y_V),reverse_normalize(ans)))
rmse


# In[69]:


# save model

import dill
import cPickle
with open("model/model_dict","wb") as f:
    dill.dump(model_dict,f)
with open('model/model_dict_3','wb') as f:
    pickle.dump(model_dict,f, protocol=pickle.HIGHEST_PROTOCOL)
model.save_weights('model/RyanCode_W.h5')
param = np.append(param_Take, [best_b])
param = np.append(param, best_w_vector)
np.save('model/model_7048.npy', param)


# ## Testing 

# In[38]:


# load model
with open('model/model_dict_2','rb') as f:
    model_dict = pickle.load(f)


# In[45]:


model_dict


# In[70]:


test_X = []
test_stat = []
test_name = []

f = open('./data/test.csv', 'r', encoding = 'big5')
# f = open(sys.argv[2], 'r', encoding = 'big5')

for line in csv.reader(f):
    
    # test number index
    if line[0] not in test_name:
        test_name.append(line[0])
        
    for stat in line[2:]:
        if stat != 'NR':
            test_stat.append(float(stat))
        # replace NR by 0
        else:
            test_stat.append(float(-1))
            
            
    
    if len(test_stat) == 9*18:
        
        
        test_X.append(test_stat)
        test_stat = []
f.close()


# In[71]:


test_X = np.asarray(test_X)
# test_X_Take = test_X[:,param_Take]
label_test_X = []


iqr = np.percentile(y, 75) - np.percentile(y, 25)
y_up_limit = np.percentile(y, 75) + 1.5 * iqr
y_down_limit = np.percentile(y, 25) - 1.5 * iqr

res_dict = {}
for k in model_dict.keys():
    res_dict[k] = model_dict[k].predict(test_X)
    

mul = np.ones((260,))
model_nums = 0
for k in res_dict.keys():
    mul *= res_dict[k]
    model_nums += 1
final_ans = mul ** (1/model_nums)

yy = []
ans = model.predict(test_X)
for s in ans:
    yy.append(s[0])
yy = np.asarray(yy)
final_ans = (final_ans*5 + yy)/6

print(final_ans)
    
result = [['id','value']]

# print(type(final_ans))

for i, j in zip(test_name, final_ans):
    line = []
    line.append(i)
    line.append(j)
    result.append(line)


# In[59]:


ans = model.predict(test_X)
print(type(final_ans))
yy = []
for s in ans:
    yy.append(s[0])
yy = np.asarray(yy)
final_ans = (final_ans*5 + ans)/6
final_ans
    print(s[0])
ans[i][0]


# In[72]:


f = open('./result/res_ens2.csv', 'w', encoding = 'big5')
# f = open(sys.argv[3], 'w', encoding = 'big5')
w = csv.writer(f)
w.writerows(result)
f.close()

