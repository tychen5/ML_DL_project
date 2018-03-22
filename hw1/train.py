
# coding: utf-8

# In[93]:


import csv
import sys
import numpy as np
import pandas as pd
# import re


# *** Train Data ***

# In[94]:


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


# In[95]:


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


# In[96]:


# df = pd.DataFrame(X_T,columns=None)
# df
X_V = X_T
y_V = y_T


# *** split into valid and testing set ***

# In[99]:


X = X_T[y_T >= 0]
X_V = X_V[y_V>=0]
y = y_T[y_T >= 0]
y_V = y_V[y_V>=0]

record = []
np.random.seed(seed = 589457)


train_X, valid_X = X , X_V
train_y, valid_y = y,y_V


train_volume = len(train_y)


# *** feature & Parameters ***

# In[134]:


feature_Take = [
        False,  False, False,  False,  False, False, False, False, False,
        True,  False, False,  False,  False, False, False, False, False]
# feature_Take = [
#         True,  True, True,  True,  True, True, True, True, True,
#         True,  True, True,  True,  True, True, True, True, True]  ##ref1_best


# time_Take = [False, False, True, True, True, True, True, True, True]
time_Take = [True, True, True, True, True, True, True, True, True]
timeSum = sum(time_Take)


# 總共 18 feature * 9 hr = 162 parameters
param_Take = np.repeat(feature_Take, 9) * np.tile(time_Take, 18)
param_Take_len = len(param_Take)

lamb = 0.99  #1e-6, 0.001, 0.1,0.99
max_iteration = 1000


initial_b = 1
initial_w = -0.01  
eps = 1e-6
lrr = 1  #(1000rpoch=1，3000epoch=0.9=>1.7 ; 0.995) #1.7
decay = 0.995
b = initial_b
b_lr = 0.0

w_vector = np.zeros(X.shape[1], dtype = np.float) + initial_w

#pm2.5 should be very relative
w_vector[81:90] = np.zeros(9, dtype = np.float) + 1/timeSum

w_lr_vector = np.zeros(X.shape[1], dtype = np.float)




x_Take = train_X[:, param_Take]

w_vector = w_vector[param_Take]
w_lr_vector = w_lr_vector[param_Take]


# In[135]:


train_mse = float("inf")
valid_mse = float("inf")
last_mseV = 0.0

# Iteration training
for i in range(max_iteration):

    b_grad = 0.0
    lr=1
    w_grad_vector = np.zeros(X.shape[1], dtype=np.float)
    w_grad_vector = w_grad_vector[param_Take]

    for n in range(len(train_y)):
        x = x_Take[n]

        e = train_y[n] - b - sum(w_vector * x) 
        e += lamb * sum(w_vector ** 2)

        b_grad = b_grad - 2.0 * e * 1.0
        w_grad_vector = w_grad_vector - 2.0 * e * x

    # Adagrad
    b_lr += b_grad ** 2
    w_lr_vector += w_grad_vector ** 2

    # Update parameters.
    b -= lr * b_grad / (np.sqrt(b_lr))
    w_vector -= lr * w_grad_vector / (np.sqrt(w_lr_vector))
    

    if (i + 1) % 10 == 0:

        train_error = []
        for y_index in range(len(train_y)):
            x = x_Take[y_index]
            train_error.append((train_y[y_index] - b - sum(w_vector * x)) ** 2)

            ###validation
        valid_error = []
        for y_index in range(len(valid_y)):
            x = valid_X[:, param_Take][y_index]
            mseV = (valid_y[y_index] - b - sum(w_vector * x)) ** 2
            valid_error.append(mseV)
#             mseV = np.mean(valid_error) ** 0.5

        if (np.mean(valid_error) > valid_mse or round(mseV,5) == round(last_mseV,5)):
            break

        else:
            train_mse = np.mean(train_error)
            valid_mse = np.mean(valid_error)
            last_mseV = mseV

        print(
            "Iteration Times: %d"
            % (i + 1),
            "train RMSE: %.5f"
            % np.mean(train_error) ** 0.5,
            "valid RMSE: %.5f"
            % np.mean(valid_error) ** 0.5
        )

best_b = b
best_w_vector = w_vector
best_valid_mse = valid_mse



train_error = []
for y_index in range(len(train_y)):
    x = x_Take[y_index]
    train_error.append((train_y[y_index] - b - sum(w_vector * x)) ** 2)

print("train_MSE: %.5f"
      % np.mean(train_error))

print("train_RMSE: %.5f"
      % np.mean(train_error) ** 0.5)

# outliers
train_error = np.sqrt(train_error)
iqr = np.percentile(train_error, 75) - np.percentile(train_error, 25)
up_limit = np.percentile(train_error, 75) + 1.5 * iqr
down_limit = np.percentile(train_error, 25) - 1.5 * iqr

while valid_mse <= best_valid_mse+eps :  #加入條件count 機制
    #     while(sum(train_error >= up_limit) > len(train_y)*0.01):  ##testing

    print("outliers count: %d"
          % sum(train_error >= up_limit))

    best_b = b
    best_w_vector = w_vector
    best_valid_mse = valid_mse

    # remove upper outliers
    train_X = train_X[(train_error < up_limit)]
    train_y = train_y[(train_error < up_limit)]

    # re-train

    initial_b = 1
    initial_w = -0.01

    lr = lrr

    b = initial_b
    b_lr = 0.0

    w_vector = np.zeros(X.shape[1], dtype=np.float) + initial_w

    # pm2.5 should be very relative
    w_vector[81:90] = np.zeros(9, dtype=np.float) + 1 / timeSum

    w_lr_vector = np.zeros(X.shape[1], dtype=np.float)

    x_Take = train_X[:, param_Take]

    w_vector = w_vector[param_Take]
    w_lr_vector = w_lr_vector[param_Take]

    train_mse = float("inf")
    valid_mse = float("inf")
    last_mseV = 0.0

    # Iterations
    for i in range(max_iteration):

        b_grad = 0.0
        w_grad_vector = np.zeros(X.shape[1], dtype=np.float)
        w_grad_vector = w_grad_vector[param_Take]

        for n in range(len(train_y)):
            x = x_Take[n]

            e = train_y[n] - b - sum(w_vector * x)
            e += lamb * sum(w_vector ** 2)

            b_grad = b_grad - 2.0 * e * 1.0
            w_grad_vector = w_grad_vector - 2.0 * e * x

        # Adagrad
        b_lr += b_grad ** 2
        w_lr_vector += w_grad_vector ** 2

        # Update parameters.
        b -= lr * b_grad / (np.sqrt(b_lr))
        w_vector -= lr * w_grad_vector / (np.sqrt(w_lr_vector))
#         if lr > 0.5:
#             lr *= decay

        if (i + 1) % 10 == 0:
            if lr > 0.5:
                lr *= decay

            train_error = []
            for y_index in range(len(train_y)):
                x = x_Take[y_index]
                train_error.append((train_y[y_index] - b - sum(w_vector * x)) ** 2)

            valid_error = []
            for y_index in range(len(valid_y)):
                x = valid_X[:, param_Take][y_index]
                mseV = (valid_y[y_index] - b - sum(w_vector * x)) ** 2
                valid_error.append(mseV)
#                 mseV = np.mean(valid_error) ** 0.5

            if (np.mean(valid_error) > valid_mse or round(mseV,5) == round(last_mseV,5)):  #round
                break

            else:
                train_mse = np.mean(train_error)
                valid_mse = np.mean(valid_error)
                last_mseV = mseV

            print(
                "Iteration Times: %d"
                % (i + 1),
                "train RMSE: %.5f"
                % np.mean(train_error) ** 0.5,
                "valid RMSE: %.5f"
                % np.mean(valid_error) ** 0.5
            )

    train_error = []
    for y_index in range(len(train_y)):
        x = x_Take[y_index]
        train_error.append((train_y[y_index] - b - sum(w_vector * x)) ** 2)

    # outliers
    train_error = np.sqrt(train_error)
    iqr = np.percentile(train_error, 75) - np.percentile(train_error, 25)
    up_limit = np.percentile(train_error, 75) + 1.5 * iqr
    down_limit = np.percentile(train_error, 25) - 1.5 * iqr

best_b = b
best_w_vector = w_vector
best_valid_mse = valid_mse


# In[ ]:


# w = w - lr * (gra/ada + 2*lambda_*w) ##l2 regularization GRADIENT
# cost_a  = math.sqrt(cost) + lambda_ * np.linalg.norm(w)/len(x)


# In[103]:


# save model

# param = np.append(param_Take, [best_b])
# param = np.append(param, best_w_vector)
# np.save('model/param_Take.npy',param_Take)
# np.save('model/model_final.npy', param)


# ## Testing 

# In[79]:


# param


# In[81]:


# load model

# param = np.load('model/model_7048.npy')
# param_Take = np.asarray(param[0:param_Take_len], dtype=bool)
# b = param[param_Take_len]
# w_vector = param[param_Take_len + 1:]


# In[136]:


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


# In[137]:


test_X = np.asarray(test_X)
test_X_Take = test_X[:,param_Take]
label_test_X = []


iqr = np.percentile(y, 75) - np.percentile(y, 25)
y_up_limit = np.percentile(y, 75) + 1.5 * iqr
y_down_limit = np.percentile(y, 25) - 1.5 * iqr

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


# In[138]:


f = open('./result/3-0.99.csv', 'w', encoding = 'big5')
# f = open(sys.argv[3], 'w', encoding = 'big5')
w = csv.writer(f)
w.writerows(result)
f.close()

