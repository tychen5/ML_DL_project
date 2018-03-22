
# coding: utf-8

# In[8]:


#DONE


# In[1]:


import csv
import sys
import numpy as np
# import pandas as pd
# import re


# ## Testing 

# In[4]:


# load model
param = np.load('models/model_final.npy')
param_Take = np.load('models/param_Take.npy')
param_Take_len = len(param_Take)
param_Take = np.asarray(param[0:param_Take_len], dtype=bool)
b = param[param_Take_len]
w_vector = param[param_Take_len + 1:]


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
test_X_Take = test_X[:,param_Take]
label_test_X = []



for stat in test_X_Take:
    x = stat
    
    if b + sum(w_vector * x) > 0:
        label_test_X.append(b + sum(w_vector * x))
    else:
        label_test_X.append(0)
    
result = [['id','value']]
for i, j in zip(test_name, label_test_X):
    line = []
    line.append(i)
    line.append(j)
    result.append(line)


# In[7]:


# f = open('./result/param_reproduce.csv', 'w', encoding = 'big5')
f = open(sys.argv[2], 'w', encoding = 'big5')
w = csv.writer(f)
w.writerows(result)
f.close()

