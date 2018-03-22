
# coding: utf-8

# In[ ]:


#DONE


# In[3]:


try:    
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
except:
    pass
# import re


# ## Testing 

# In[4]:


# load model
with open('models/model_dict_2','rb') as f:
    model_dict = pickle.load(f)


# In[5]:


test_X = []
test_stat = []
test_name = []

# f = open('./data/test.csv', 'r', encoding = 'big5')
f = open(sys.argv[1], 'r', encoding = 'big5')

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


# In[6]:


test_X = np.asarray(test_X)
label_test_X = []

res_dict = {}
for k in model_dict.keys():
    res_dict[k] = model_dict[k].predict(test_X)
    

mul = np.ones((260,))
model_nums = 0
for k in res_dict.keys():
    mul *= res_dict[k]
    model_nums += 1
final_ans = mul ** (1/model_nums)

    
result = [['id','value']]

for i, j in zip(test_name, final_ans):
    line = []
    line.append(i)
    line.append(j)
    result.append(line)


# In[7]:


# f = open('./result/reproduce_best.csv', 'w', encoding = 'big5')
f = open(sys.argv[2], 'w', encoding = 'big5')
w = csv.writer(f)
w.writerows(result)
f.close()

