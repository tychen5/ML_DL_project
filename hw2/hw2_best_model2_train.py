
# coding: utf-8

# In[205]:


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
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
import time
import pickle
from sklearn.linear_model import *
from sklearn.kernel_ridge import *


# In[348]:


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


# In[4]:


def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
#     print(X.shape, Y.shape)
    return (X[randomize], Y[randomize])


# In[5]:


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


# In[6]:


def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid


# In[7]:


def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))


# In[8]:


def valid(w, b, X_valid, Y_valid):
    valid_data_size = len(X_valid)

    z = (np.dot(X_valid, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    print('Validation acc = %f' % (float(result.sum()) / valid_data_size))
    return


# In[9]:


def train(X_all, Y_all, save_dir):
    # Split a 10%-validation set from the training set
    valid_set_percentage = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)
    print(len(X_train),len(X_valid))

    # Initiallize parameter, hyperparameter
    w = np.zeros((123,)) #106
    b = np.zeros((1,))
    l_rate = 0.1
    batch_size = 32
    train_data_size = len(X_train)
    step_num = int(floor(train_data_size / batch_size))
    epoch_num = 3000
    save_param_iter = 50

    # Start training
    total_loss = 0.0
    for epoch in range(1, epoch_num):
        # Do validation and parameter saving
        if (epoch) % save_param_iter == 0:
            print('=====Saving Param at epoch %d=====' % epoch)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            np.savetxt(os.path.join(save_dir, 'w'), w)
            np.savetxt(os.path.join(save_dir, 'b'), [b,])
            print('epoch avg loss = %f' % (total_loss / (float(save_param_iter) * train_data_size)))
            total_loss = 0.0
            valid(w, b, X_valid, Y_valid)

        # Random shuffle
        X_train, Y_train = _shuffle(X_train, Y_train)

        # Train with batch
        for idx in range(step_num):
            X = X_train[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]

            z = np.dot(X, np.transpose(w)) + b
            y = sigmoid(z)

            cross_entropy = -1 * (np.dot(np.squeeze(Y), np.log(y)) + np.dot((1 - np.squeeze(Y)), np.log(1 - y)))
            total_loss += cross_entropy

            w_grad = np.mean(-1 * X * (np.squeeze(Y) - y).reshape((batch_size,1)), axis=0)
            b_grad = np.mean(-1 * (np.squeeze(Y) - y))

            # SGD updating parameters
            w = w - l_rate * w_grad
            b = b - l_rate * b_grad

    return


# ## Predict Testing data

# In[19]:


def infer(X_test, save_dir, output_dir):
    test_data_size = len(X_test)

    # Load parameters
    print('=====Loading Param from %s=====' % save_dir)
    w = np.loadtxt(os.path.join(save_dir, 'w'))
    b = np.loadtxt(os.path.join(save_dir, 'b'))

    # predict
    z = (np.dot(X_test, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)

    print('=====Write output to %s =====' % output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, 'Sample_Logistic.csv')
    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_):
            f.write('%d,%d\n' %(i+1, v))

    return


# ## Params

# In[55]:


def main(opts):
    # Load feature and label
#     X_all, Y_all, X_test = load_data('data/data_old/X_train', 'data/data_old/Y_train', 'data/data_old/X_test')
    X_all, Y_all, X_test = load_data('data/train_X', 'data/train_Y', 'data/test_X')
    # Normalization
    X_all, X_test = normalize(X_all, X_test)
    
    trains = True
    infers = False
    # To train or to infer
    if trains:
#         train(X_all, Y_all, 'model/model_old/')
        train(X_all, Y_all, 'model/')
    elif infers:
#         infer(X_test,  'model/model_old/', 'result/result_old/')
        infer(X_test,  'model/', 'result/')
    else:
        print("Error: Argument --train or --infer not found")
    return

if __name__ == '__main__':
    opts = 'kk'
#     parser = argparse.ArgumentParser(description='Logistic Regression with Gradient Descent Method')
#     group = parser.add_mutually_exclusive_group()
#     group.add_argument('--train', action='store_true', default=True,
#                         dest='train', help='Input --train to Train')
#     group.add_argument('--infer', action='store_true',default=False,
#                         dest='infer', help='Input --infer to Infer')
#     parser.add_argument('--train_data_path', type=str,
#                         default='data/data_old/X_train', dest='train_data_path',
#                         help='Path to training data')
#     parser.add_argument('--train_label_path', type=str,
#                         default='data/data_old/X_train', dest='train_label_path',
#                         help='Path to training data\'s label')
#     parser.add_argument('--test_data_path', type=str,
#                         default='data/X_test', dest='test_data_path',
#                         help='Path to testing data')
#     parser.add_argument('--save_dir', type=str,
#                         default='model/', dest='save_dir',
#                         help='Path to save the model parameters')
#     parser.add_argument('--output_dir', type=str,
#                         default='model/', dest='output_dir',
#                         help='Path to save the model parameters')
#     opts = parser.parse_args()
    main(opts)


# In[60]:


X_train = pd.read_csv("data/train_X", sep=',', header=0)
X_train


# In[68]:


# for i in X_train.columns:
#     print(i)
pd.set_option('display.max_columns', None)
X_train.head(100)


# # SKLEARN ENSEMBLE

# 拿0.75去train，0.25 test，算出來最好的ACC，當成權重去乘以每個predict出來的label。這個權重也乘為分母ensemble的加項
# 
# 去跟NN ensemble，不用轉乘label
# 
# <0.5=>0，else:1

# In[373]:


with open('model/reg_col.picke', 'rb') as f:
    reg_col = pickle.load(f)
len(reg_col)


# In[564]:


X_all, Y_all, X_test = load_data('data/train_X', 'data/train_Y', 'data/test_X',reg_col)
X_all, X_test = normalize(X_all, X_test)
valid_set_percentage=0.999
X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)
Y_train = Y_train.ravel()
print(X_train.shape, Y_train.shape)
print(X_valid.shape,Y_valid.shape)


# ## KNeighborsRegrssor
# done

# In[565]:


# start_time = time.time()
# neigh = KNeighborsClassifier(algorithm='auto',weights='uniform',p=1,n_jobs=11) #n_jobs=10
neigh = KNeighborsRegressor(algorithm='auto',p=1,weights='uniform',n_jobs=-1,n_neighbors=33,leaf_size=45) #n_jobs=10
neigh.fit(X_train, Y_train) 
# print("--- %s seconds ---" % (time.time() - start_time))


# In[566]:


with open('model/neigh_r.pickle', 'wb') as f:
    pickle.dump(neigh, f)


# In[481]:


start_time = time.time()
parameters = {"n_neighbors":[33,32,34],'leaf_size':[45,44,46]} #n_neigh=5,10 , leaf_size=20
#'weights':['uniform','distance'] ,'p':[1,2,3]
clf = GridSearchCV(neigh,parameters,verbose=2,n_jobs=11,scoring='neg_mean_squared_error')
clf.fit(X_train, Y_train)
print("--- %s seconds ---" % (time.time() - start_time))


# In[482]:


clf.grid_scores_, clf.best_params_, clf.best_score_


# In[509]:


start_time = time.time()
y = neigh.predict(X_valid)
# y1 = neigh.predict(X_test)
print("--- %s seconds ---" % (time.time() - start_time))
# y


# In[510]:


ans = []
for i,v in enumerate(y):
    if v < 0.5:
        vv=0
        ans.append(abs(vv-Y_valid[i][0]))
    else:
        vv=1
        ans.append(abs(vv-Y_valid[i][0]))
#     print(i)
acc1 = ans.count(0)/len(ans)
print("ACC:",acc1)


# In[16]:


# Y_valid[2][0]
# ACC: 0.8223552894211577
#0.8273136773136773


# ## SVR
# done

# In[567]:


start_time = time.time()
svr = SVR(kernel='rbf',shrinking=True, gamma='auto',C=0.7)
# svr = SVR(kernel='rbf')
svr.fit(X_train, Y_train)
# print("--- %s seconds ---" % (time.time() - start_time))


# In[568]:


with open('model/svr_r.pickle', 'wb') as f:
    pickle.dump(svr, f)


# In[451]:


start_time = time.time()
parameters = {"C":[1,3,5]} #probability:[True,False],shrinking:[True,False],decision_function_shape:['ovo','ovr'] # #,'gamma':[2,'auto'],'shrinking':[True,False]
#
clf2 = GridSearchCV(svr,parameters,verbose=2,n_jobs=11,scoring='neg_mean_squared_error')
clf2.fit(X_train, Y_train)
print("--- %s seconds ---" % (time.time() - start_time))


# In[452]:


clf2.grid_scores_, clf2.best_params_, clf2.best_score_
#{'kernel': 'rbf'},0.3576666151408195)
#-0.XX


# In[518]:


start_time = time.time()
y= svr.predict(X_valid)
# y2 = svr.predict(X_test)
print("--- %s seconds ---" % (time.time() - start_time))
# y


# In[54]:


c=0
for ii in y:
    if ii < 0:
#         print("<0")
        print(ii)
        c+=1
    elif ii>1:
#         print(">1")
        print(ii)
        c+=1
print(c/len(y))


# In[519]:


ans = []
for i,v in enumerate(y):
    if v < 0.5:
        vv=0
        ans.append(abs(vv-Y_valid[i][0]))
    else:
        vv=1
        ans.append(abs(vv-Y_valid[i][0]))
#     print(i)
acc2 = ans.count(0)/len(ans)
print("ACC:",acc2)
#ACC 0.8517=>
#ACC: 0.8481758997666135


# ## Decision Tree
# ### done

# In[569]:


start_time = time.time()
# dtc = DecisionTreeClassifier()
dtr = DecisionTreeRegressor(max_features='log2',splitter='best',presort=False,min_samples_split=11,min_samples_leaf=19)
dtr.fit(X_train,Y_train)
# print("--- %s seconds ---" % (time.time() - start_time))


# In[570]:


with open('model/dtr_r.pickle', 'wb') as f:
    pickle.dump(dtr, f)


# In[475]:


start_time = time.time()
parameters = {'min_samples_leaf':[20,19]}
# ,'max_features':[1,10,'auto','sqrt','log2',None] ,'presort':[True,False],'splitter':['best','random']'min_samples_split':[11,12]
clf4 = GridSearchCV(dtr,parameters,verbose=2,n_jobs=-1,scoring='neg_mean_squared_error')
clf4.fit(X_train, Y_train)
print("--- %s seconds ---" % (time.time() - start_time))


# In[476]:


clf4.grid_scores_, clf4.best_params_, clf4.best_score_


# In[521]:


start_time = time.time()
y = dtr.predict(X_valid)
# y4 = dtr.predict(X_test)
print("--- %s seconds ---" % (time.time() - start_time))
# y


# In[522]:


ans = []
for i,v in enumerate(y):
    if v < 0.5:
        vv=0
        ans.append(abs(vv-Y_valid[i][0]))
    else:
        vv=1
        ans.append(abs(vv-Y_valid[i][0]))
#     print(i)
acc4 = ans.count(0)/len(ans)
print("ACC:",acc4)


# ## MLP
# done

# In[571]:


start_time = time.time()
mlp = MLPRegressor(max_iter=1000,nesterovs_momentum=False,learning_rate='invscaling',solver='adam',activation='logistic',
                  hidden_layer_sizes=(128,))
# mlp = MLPRegressor()
mlp.fit(X_train,Y_train)
# print("--- %s seconds ---" % (time.time() - start_time))


# In[572]:


with open('model/mlp_r.pickle', 'wb') as f:
    pickle.dump(mlp, f)


# In[484]:


start_time = time.time()
parameters = {'activation':['logistic','tanh','relu'],'solver':['lbfgs','sgd','adam'],'learning_rate':['invscaling','adaptive'],'nesterovs_momentum':[True,False],'warm_start':[True,False],'early_stopping':[True,False]}
#neg_mean_squared_error
#,
#'hidden_layer_sizes':[(256,128,64,32,16,8),(256,256,256,256,256)],
clf5 = GridSearchCV(mlp,parameters,verbose=2,n_jobs=11,scoring='neg_mean_squared_error')
clf5.fit(X_train, Y_train)
print("--- %s seconds ---" % (time.time() - start_time))


# In[485]:


clf5.grid_scores_, clf5.best_params_, clf5.best_score_


# In[530]:


start_time = time.time()
y = mlp.predict(X_valid)
# y5 = mlp.predict(X_test)
print("--- %s seconds ---" % (time.time() - start_time))
# y


# In[531]:


ans = []
for i,v in enumerate(y):
    if v < 0.5:
        vv=0
        ans.append(abs(vv-Y_valid[i][0]))
    else:
        vv=1
        ans.append(abs(vv-Y_valid[i][0]))
#     print(i)
acc5 = ans.count(0)/len(ans)
print("ACC:",acc5)


# ## Logistic Regression
# done

# In[573]:


start_time = time.time()
LR = LogisticRegression(max_iter=1000,class_weight=None,solver='sag',fit_intercept=True,multi_class='ovr',warm_start=False,C=0.7) #n_jobs=11
# LR = LogisticRegression() #n_jobs=11
LR.fit(X_train,Y_train)
# print("--- %s seconds ---" % (time.time() - start_time))


# In[574]:


with open('model/LR_r.pickle', 'wb') as f:
    pickle.dump(LR, f)


# In[499]:


start_time = time.time()
parameters = {'multi_class':['ovr', 'multinomial']
             ,'warm_start':[True,False]}
#,'C':[0.9,1,2],'class_weight':['balanced',None],'solver':[ 'lbfgs', 'liblinear', 'sag', 'saga']'fit_intercept':[True,False]
#'penalty':['l1','l2'],'dual':[True,False],,'multi_class':['ovr', 'multinomial']
clf6 = GridSearchCV(LR,parameters,verbose=2,n_jobs=-1,scoring='neg_mean_squared_error')
clf6.fit(X_train, Y_train)
print("--- %s seconds ---" % (time.time() - start_time))


# In[500]:


clf6.grid_scores_, clf6.best_params_, clf6.best_score_


# In[539]:


start_time = time.time()
y = LR.predict(X_valid)
print("--- %s seconds ---" % (time.time() - start_time))
y


# In[540]:


ans = []
for i,v in enumerate(y):
    if v < 0.5:
        vv=0
        ans.append(abs(vv-Y_valid[i][0]))
    else:
        vv=1
        ans.append(abs(vv-Y_valid[i][0]))
#     print(i)
acc6 = ans.count(0)/len(ans)
print("ACC:",acc6)


# ## Linear SVR
# done

# In[575]:


# start_time = time.time()
lsvr = LinearSVR(fit_intercept=True,loss='squared_epsilon_insensitive',dual=False,C=0.8)
lsvr.fit(X_train,Y_train)
# print("--- %s seconds ---" % (time.time() - start_time))


# In[576]:


with open('model/lsvr_r.pickle', 'wb') as f:
    pickle.dump(lsvr, f)


# In[505]:


start_time = time.time()
parameters = {'dual':[True,False]}
#,'dual':[True,False]
#C:[0.8,1,1,2]
clf7 = GridSearchCV(lsvr,parameters,verbose=2,n_jobs=-1,scoring='neg_mean_squared_error')
clf7.fit(X_train, Y_train)
print("--- %s seconds ---" % (time.time() - start_time))


# In[506]:


clf7.grid_scores_, clf7.best_params_, clf7.best_score_


# In[542]:


start_time = time.time()
y = lsvr.predict(X_valid)
# y7 = lsvr.predict(X_test)
print("--- %s seconds ---" % (time.time() - start_time))
# y


# In[543]:


ans = []
for i,v in enumerate(y):
    if v < 0.5:
        vv=0
        ans.append(abs(vv-Y_valid[i][0]))
    else:
        vv=1
        ans.append(abs(vv-Y_valid[i][0]))
#     print(i)
acc7 = ans.count(0)/len(ans)
print("ACC:",acc7)


# ## Ridge

# In[577]:


reg = linear_model.Ridge()
reg.fit(X_train,Y_train)


# In[578]:


with open('model/reg_r.pickle', 'wb') as f:
    pickle.dump(reg, f)


# In[545]:


start_time = time.time()
y8 = reg.predict(X_valid)
print("--- %s seconds ---" % (time.time() - start_time))
y8


# In[546]:


ans = []
for i,v in enumerate(y8):
    if v < 0.5:
        vv=0
        ans.append(abs(vv-Y_valid[i][0]))
    else:
        vv=1
        ans.append(abs(vv-Y_valid[i][0]))
#     print(i)
acc8 = ans.count(0)/len(ans)
print("ACC:",acc8)


# ## Lasso

# In[579]:


las = linear_model.Lasso(alpha=0.0001)#alpha=0.0001
las.fit(X_train,Y_train)


# In[580]:


with open('model/las_r.pickle', 'wb') as f:
    pickle.dump(las, f)


# In[548]:


start_time = time.time()
y9 = las.predict(X_valid)
print("--- %s seconds ---" % (time.time() - start_time))
y9


# In[549]:


ans = []
for i,v in enumerate(y9):
    if v < 0.5:
        vv=0
        ans.append(abs(vv-Y_valid[i][0]))
    else:
        vv=1
        ans.append(abs(vv-Y_valid[i][0]))
#     print(i)
acc9 = ans.count(0)/len(ans)
print("ACC:",acc9)


# ## ElasticNet

# In[581]:


en = linear_model.ElasticNet(alpha=0.0001,max_iter=2000)
en.fit(X_train,Y_train)


# In[582]:


with open('model/en_r.pickle', 'wb') as f:
    pickle.dump(en, f)


# In[551]:


start_time = time.time()
y10 = en.predict(X_valid)
print("--- %s seconds ---" % (time.time() - start_time))
y10


# In[552]:


ans = []
for i,v in enumerate(y10):
    if v < 0.5:
        vv=0
        ans.append(abs(vv-Y_valid[i][0]))
    else:
        vv=1
        ans.append(abs(vv-Y_valid[i][0]))
#     print(i)
acc10 = ans.count(0)/len(ans)
print("ACC:",acc10)


# ## OMG

# In[583]:


omg = OrthogonalMatchingPursuit()
omg.fit(X_train,Y_train)


# In[584]:


with open('model/omg_r.pickle', 'wb') as f:
    pickle.dump(omg, f)


# In[554]:


start_time = time.time()
y11 = omg.predict(X_valid)
print("--- %s seconds ---" % (time.time() - start_time))
y11


# In[555]:


ans = []
for i,v in enumerate(y11):
    if v < 0.5:
        vv=0
        ans.append(abs(vv-Y_valid[i][0]))
    else:
        vv=1
        ans.append(abs(vv-Y_valid[i][0]))
#     print(i)
acc11 = ans.count(0)/len(ans)
print("ACC:",acc11)


# ## BayesianRidge

# In[585]:


br = linear_model.BayesianRidge()
br.fit(X_train,Y_train)


# In[586]:


with open('model/br_r.pickle', 'wb') as f:
    pickle.dump(br, f)


# In[557]:


start_time = time.time()
y12 = br.predict(X_valid)
print("--- %s seconds ---" % (time.time() - start_time))
y12


# In[558]:


ans = []
for i,v in enumerate(y12):
    if v < 0.5:
        vv=0
        ans.append(abs(vv-Y_valid[i][0]))
    else:
        vv=1
        ans.append(abs(vv-Y_valid[i][0]))
#     print(i)
acc12 = ans.count(0)/len(ans)
print("ACC:",acc12)


# ## ARDR

# In[559]:


ardr = linear_model.ARDRegression(verbose=True)  #很久，沒n_jobs
ardr.fit(X_train,Y_train)


# In[443]:


with open('model/ardr_r.pickle', 'wb') as f:
    pickle.dump(ardr, f)


# In[412]:


start_time = time.time()
y13 = ardr.predict(X_valid)
print("--- %s seconds ---" % (time.time() - start_time))
y13


# In[413]:


ans = []
for i,v in enumerate(y13):
    if v < 0.5:
        vv=0
        ans.append(abs(vv-Y_valid[i][0]))
    else:
        vv=1
        ans.append(abs(vv-Y_valid[i][0]))
#     print(i)
acc13 = ans.count(0)/len(ans)
print("ACC:",acc13)
#ACC: 0.8388404372927158


# ## TheilSenRegressor

# In[587]:


tsr =  linear_model.TheilSenRegressor()
tsr.fit(X_train,Y_train)


# In[588]:


with open('model/tsr_r.pickle', 'wb') as f:
    pickle.dump(tsr, f)


# In[561]:


start_time = time.time()
y14 = tsr.predict(X_valid)
print("--- %s seconds ---" % (time.time() - start_time))
y14


# In[562]:


ans = []
for i,v in enumerate(y14):
    if v < 0.5:
        vv=0
        ans.append(abs(vv-Y_valid[i][0]))
    else:
        vv=1
        ans.append(abs(vv-Y_valid[i][0]))
#     print(i)
acc14 = ans.count(0)/len(ans)
print("ACC:",acc14)


# ***

# In[92]:


y1[y1<0]=0
y1[y1>1]=1
y2[y2<0]=0
y2[y2>1]=1
y3[y3<0]=0
y3[y3>1]=1
y4[y4<0]=0
y4[y4>1]=1
y5[y5<0]=0
y5[y5>1]=1
y7[y7<0]=0
y7[y7>1]=1
y1 = y1*2-1
y2 = y2*2-1
y3 = y3*2-1
y4 = y4*2-1
y5 = y5*2-1
# y6 = y6*2-1
y7 = y7*2-1
ens = (y1*acc1+y2*acc2+y3*acc3+y4*acc4+y5*acc5+y7*acc7)/(acc1+acc2+acc3+acc4+acc5+acc7) #家群平均
final_ans =[]
for v in ens: #轉回label
    if v < 0:
        final_ans.append(0)
    else:
        final_ans.append(1)
final_ans


# In[93]:


test_data_size = len(X_test)

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
output_path = os.path.join(output_dir, 'ens_regressor.csv')
with open(output_path, 'w') as f:
    f.write('id,label\n')
    for i, v in  enumerate(final_ans):
        f.write('%d,%d\n' %(i+1, v))


# In[563]:


## acc_list  to pickle
acc_list=[]
acc3=0
# acc6=0
acc_list.append(acc1)
acc_list.append(acc2)
acc_list.append(acc3)
acc_list.append(acc4)
acc_list.append(acc5)
acc_list.append(acc6)
acc_list.append(acc7)
acc_list.append(acc8)
acc_list.append(acc9)
acc_list.append(acc10)
acc_list.append(acc11)
acc_list.append(acc12)
acc_list.append(acc13)
acc_list.append(acc14)

with open('model/r_acc1_7.pickle', 'wb') as f:
    pickle.dump(acc_list, f)


# ***

# ## RadiusNeighborsClassifier
# ### 有問題(?

# In[114]:


start_time = time.time()
rnc = RadiusNeighborsRegressor()
rnc.fit(X_train,Y_train)
print("--- %s seconds ---" % (time.time() - start_time))


# In[115]:


start_time = time.time()
y = rnc.predict(X_valid)
print("--- %s seconds ---" % (time.time() - start_time))
y


# In[116]:


ans = []
for i,v in enumerate(y):
    ans.append(abs(v-Y_valid[i][0]))
#     print(i)
print("ACC:",ans.count(0)/len(ans))


# ## GNB
# ### 不要用

# In[32]:


start_time = time.time()
gnb = GaussianNB()
gnb.fit(X_train,Y_train)
print("--- %s seconds ---" % (time.time() - start_time))


# In[33]:


start_time = time.time()
y = gnb.predict(X_valid)
print("--- %s seconds ---" % (time.time() - start_time))
y


# In[34]:


ans = []
for i,v in enumerate(y):
    ans.append(abs(v-Y_valid[i][0]))
#     print(i)
print("ACC:",ans.count(0)/len(ans))


# ## QDA
# ### 不要用好了

# In[35]:


start_time = time.time()
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train,Y_train)
print("--- %s seconds ---" % (time.time() - start_time))


# In[36]:


start_time = time.time()
y = qda.predict(X_valid)
print("--- %s seconds ---" % (time.time() - start_time))
y


# In[37]:


ans = []
for i,v in enumerate(y):
    ans.append(abs(v-Y_valid[i][0]))
#     print(i)
print("ACC:",ans.count(0)/len(ans))


# ## GPC
# ### 放棄
# regressor 不可用 (會有附號)
# 
# 僅可用classifier
# 
# 不要用了

# In[84]:


start_time = time.time()
# gpc = GaussianProcessClassifier()  #n_jobs=10 #太久，不是合作grid search
gpr = GaussianProcessRegressor()
gpr.fit(X_train,Y_train)
# print("--- %s seconds ---" % (time.time() - start_time))


# In[98]:


with open('model/gpr_r.pickle', 'wb') as f:
    pickle.dump(gpr, f)


# In[44]:


start_time = time.time()
# parameters = {'n_restarts_optimizer':[0,1,10],'normalize_y':[True,False],'copy_X_train':[True,False]}
parameters = {'warm_start':[True,False],''}
clf3 = GridSearchCV(gpc,parameters,verbose=2,n_jobs=11,scoring='accuracy')
clf3.fit(X_train, Y_train)
print("--- %s seconds ---" % (time.time() - start_time))


# In[39]:


clf3.grid_scores_, clf3.best_params_, clf3.best_score_


# In[85]:


start_time = time.time()
# y = gpr.predict(X_valid)
y3 = gpr.predict(X_test)
print("--- %s seconds ---" % (time.time() - start_time))
# y


# In[64]:


ans = []
for i,v in enumerate(y):
    if v < 0.5:
        vv=0
        ans.append(abs(vv-Y_valid[i][0]))
    else:
        vv=1
        ans.append(abs(vv-Y_valid[i][0]))
#     print(i)
acc3 = ans.count(0)/len(ans)
print("ACC:",acc3)


# ## LLars

# In[261]:


llars = linear_model.LassoLars(alpha=.001)
llars.fit(X_train,Y_train)


# In[262]:


start_time = time.time()
y11 = llars.predict(X_valid)
print("--- %s seconds ---" % (time.time() - start_time))
y11


# In[263]:


ans = []
for i,v in enumerate(y11):
    if v < 0.5:
        vv=0
        ans.append(abs(vv-Y_valid[i][0]))
    else:
        vv=1
        ans.append(abs(vv-Y_valid[i][0]))
#     print(i)
acc11 = ans.count(0)/len(ans)
print("ACC:",acc11)


# ## HuberRegressor

# In[258]:


hr = linear_model.HuberRegressor()
hr.fit(X_train,Y_train)


# In[259]:


start_time = time.time()
y16 = hr.predict(X_valid)
print("--- %s seconds ---" % (time.time() - start_time))
y16


# In[260]:


ans = []
for i,v in enumerate(y16):
    if v < 0.5:
        vv=0
        ans.append(abs(vv-Y_valid[i][0]))
    else:
        vv=1
        ans.append(abs(vv-Y_valid[i][0]))
#     print(i)
acc16 = ans.count(0)/len(ans)
print("ACC:",acc16)


# ## KernelRidge

# In[255]:


kr = KernelRidge(alpha=0.001)
kr.fit(X_train,Y_train)


# In[256]:


start_time = time.time()
y17 = kr.predict(X_valid)
print("--- %s seconds ---" % (time.time() - start_time))
y17


# In[257]:


ans = []
for i,v in enumerate(y17):
    if v < 0.5:
        vv=0
        ans.append(abs(vv-Y_valid[i][0]))
    else:
        vv=1
        ans.append(abs(vv-Y_valid[i][0]))
#     print(i)
acc17 = ans.count(0)/len(ans)
print("ACC:",acc17)


# ## PassiveAggressiveRegressor

# In[252]:


par = PassiveAggressiveRegressor(random_state=0)
par.fit(X_train,Y_train)


# In[253]:


start_time = time.time()
y14 = par.predict(X_valid)
print("--- %s seconds ---" % (time.time() - start_time))
y14


# In[254]:


ans = []
for i,v in enumerate(y14):
    if v < 0.5:
        vv=0
        ans.append(abs(vv-Y_valid[i][0]))
    else:
        vv=1
        ans.append(abs(vv-Y_valid[i][0]))
#     print(i)
acc14 = ans.count(0)/len(ans)
print("ACC:",acc14)

