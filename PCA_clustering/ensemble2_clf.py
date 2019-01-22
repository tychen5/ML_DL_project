
# coding: utf-8

# In[469]:


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
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import *
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GridSearchCV
import time
import pickle


# In[478]:


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


# In[3]:


def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
#     print(X.shape, Y.shape)
    return (X[randomize], Y[randomize])


# In[4]:


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


# In[5]:


def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid


# In[34]:


def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))


# In[7]:


def valid(w, b, X_valid, Y_valid):
    valid_data_size = len(X_valid)

    z = (np.dot(X_valid, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    print('Validation acc = %f' % (float(result.sum()) / valid_data_size))
    return


# In[54]:


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

# In[533]:


with open('model/clf_col.pickle','rb') as f:
    clf_col = pickle.load(f)
len(clf_col)


# In[694]:


X_all, Y_all, X_test = load_data('data/train_X', 'data/train_Y', 'data/test_X',clf_col)
X_all, X_test = normalize(X_all, X_test)
valid_set_percentage=0.999
X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)
Y_train = Y_train.ravel()
print(X_train.shape, Y_train.shape)
print(X_valid.shape,Y_valid.shape)


# ## KNeighborsClassifier
# done

# In[695]:


start_time = time.time()
neigh = KNeighborsClassifier(algorithm='auto',weights='uniform',p=1,leaf_size=20,n_neighbors=33) #n_jobs=10
# neigh = KNeighborsRegressor(algorithm='auto',p=1,weights='uniform') #n_jobs=10
neigh.fit(X_train, Y_train) 
# print("--- %s seconds ---" % (time.time() - start_time))


# In[696]:


with open('model/neigh_c_2.pickle', 'wb') as f:
    pickle.dump(neigh, f)


# In[668]:


start_time = time.time()
parameters = {"n_neighbors":[33,34,32]} #n_neigh=5,10 , leaf_size=20,答案直接釘,18(20),51(20)
#,'weights':['uniform','distance'] ,'p':[1,2,3]
clf = GridSearchCV(neigh,parameters,verbose=2,n_jobs=11,scoring='accuracy')
clf.fit(X_train, Y_train)
print("--- %s seconds ---" % (time.time() - start_time))


# In[669]:


clf.grid_scores_, clf.best_params_, clf.best_score_
#{'leaf_size': 20, 'n_neighbors': 20, 'p': 1, 'weights': 'uniform'},0.3604631945512268)


# In[672]:


start_time = time.time()
y = neigh.predict(X_valid)
# y1 = neigh.predict(X_test)
# y
print("--- %s seconds ---" % (time.time() - start_time))


# In[114]:


# y


# In[673]:


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
#0.8312


# In[16]:


# Y_valid[2][0]
# ACC: 0.8223552894211577
#0.8273136773136773


# ## SVC
# done

# In[697]:


# start_time = time.time()
# svr = SVR()
svc =SVC(kernel='rbf',decision_function_shape='ovo',shrinking=True,probability=True,gamma='auto',C=0.9)
svc.fit(X_train, Y_train)
# print("--- %s seconds ---" % (time.time() - start_time))


# In[698]:


with open('model/svc_c_3.pickle', 'wb') as f:
    pickle.dump(svc, f)


# In[598]:


start_time = time.time()
parameters = {"C":[1,3,5]} #probability:[True,False],shrinking:[True,False],decision_function_shape:['ovo','ovr']
#,'decision_function_shape':['ovo','ovr'],'shrinking':[True,False],'probability':[True,False],'gamma':[2,'auto']
clf2 = GridSearchCV(svc,parameters,verbose=2,n_jobs=11,scoring='accuracy')
clf2.fit(X_train, Y_train)
print("--- %s seconds ---" % (time.time() - start_time))


# In[600]:


clf2.grid_scores_, clf2.best_params_, clf2.best_score_


# In[675]:


start_time = time.time()
y= svc.predict(X_valid)
# y2 = svc.predict(X_test)
print("--- %s seconds ---" % (time.time() - start_time))
# y


# In[676]:


ans = []
for i,v in enumerate(y):
    ans.append(abs(v-Y_valid[i][0]))
#     print(i)
acc2 = ans.count(0)/len(ans)
print("ACC:",acc2)
#ACC: 0.8516812528788577
#ACC: 0.8530893010686648


# ## Decision Tree
# done

# In[699]:


# start_time = time.time()
dtc = DecisionTreeClassifier(max_features='sqrt',presort=True,splitter='best',min_samples_split=18,min_samples_leaf=10)
# dtr = DecisionTreeRegressor()
dtc.fit(X_train,Y_train)
# print("--- %s seconds ---" % (time.time() - start_time))


# In[700]:


with open('model/dtc_c_1.5.pickle', 'wb') as f:
    pickle.dump(dtc, f)


# In[627]:


start_time = time.time()
parameters = {'min_samples_leaf':[10,9,11,12]
              }
#'max_features':[1,20,'auto','sqrt','log2',None] 'splitter':['best','random'],presort:[True,False]'min_samples_split':[17,19,18]
clf4 = GridSearchCV(dtc,parameters,verbose=2,n_jobs=11,scoring='accuracy')
clf4.fit(X_train, Y_train)
print("--- %s seconds ---" % (time.time() - start_time))


# In[628]:


clf4.grid_scores_, clf4.best_params_, clf4.best_score_


# In[678]:


start_time = time.time()
y = dtc.predict(X_valid)
# y4 = dtc.predict(X_test)
print("--- %s seconds ---" % (time.time() - start_time))
# y


# In[679]:


ans = []
for i,v in enumerate(y):
    ans.append(abs(v-Y_valid[i][0]))
#     print(i)
acc4 = ans.count(0)/len(ans)
print("ACC:", acc4)
#0.812


# ## MLP
# done

# In[701]:


# start_time = time.time()
# mlp = MLPRegressor(max_iter=1000)
mlp = MLPClassifier(max_iter=1000,activation='logistic',solver='adam',learning_rate='adaptive',nesterovs_momentum=True,warm_start=True
                   ,hidden_layer_sizes=(256, 128, 64, 32, 16, 8))
mlp.fit(X_train,Y_train)
# print("--- %s seconds ---" % (time.time() - start_time))


# In[702]:


with open('model/mlp_c_2.pickle', 'wb') as f:
    pickle.dump(mlp, f)


# In[604]:


start_time = time.time()
parameters = {'hidden_layer_sizes':[(256,128,64,32,16,8),(256,256,256,256,256)],'warm_start':[True,False],'early_stopping':[True,False]}
#'activation':['logistic','tanh','relu'],'solver':['lbfgs','sgd','adam'],'learning_rate':['invscaling','adaptive'],,'nesterovs_momentum':[True,False]
#'hidden_layer_sizes':[(256,128,64,32,16,8),(256,256,256,256,256)],
clf5 = GridSearchCV(mlp,parameters,verbose=2,n_jobs=11,scoring='accuracy')
clf5.fit(X_train, Y_train)
print("--- %s seconds ---" % (time.time() - start_time))


# In[607]:


clf5.grid_scores_, clf5.best_params_, clf5.best_score_
#  {'activation': 'logistic',
#   'early_stopping': False,
#   'learning_rate': 'adaptive',
#   'nesterovs_momentum': True,
#   'solver': 'adam',
#   'warm_start': False},
#  -0.10411238218157795)


# In[682]:


start_time = time.time()
y = mlp.predict(X_valid)
# y5 = mlp.predict(X_test)
print("--- %s seconds ---" % (time.time() - start_time))
# y


# In[683]:


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

# In[703]:


start_time = time.time()
LR = LogisticRegression(max_iter=1000,class_weight=None,solver='saga',fit_intercept=True,warm_start=True,
                        multi_class='ovr',penalty='l1',C=0.8) #n_jobs=11
LR.fit(X_train,Y_train)
# print("--- %s seconds ---" % (time.time() - start_time))


# In[704]:


start_time = time.time()
parameters = {'C':[0.9,1,2]} #'class_weight':['balanced',None ,'solver':[ 'lbfgs', 'liblinear', 'sag', 'saga']'fit_intercept':[True,False],
#,'C':[0.9,1,2],'dual':[True,False],'multi_class':['ovr', 'multinomial']
#'penalty':['l1','l2']
clf6 = GridSearchCV(LR,parameters,verbose=2,n_jobs=-1,scoring='accuracy')
clf6.fit(X_train, Y_train)
print("--- %s seconds ---" % (time.time() - start_time))


# In[664]:


clf6.grid_scores_, clf6.best_params_, clf6.best_score_


# In[705]:


with open('model/LR_c_3.pickle', 'wb') as f:
    pickle.dump(LR, f)


# In[685]:


start_time = time.time()
y = LR.predict(X_valid)
# y6 = LR.predict(X_test)
print("--- %s seconds ---" % (time.time() - start_time))
# y


# In[686]:


ans = []
for i,v in enumerate(y):
    ans.append(abs(v-Y_valid[i][0]))
#     print(i)
acc6 = ans.count(0)/len(ans)
print("ACC:",acc6)


# ## Linear SVC
# done=>??

# In[706]:


start_time = time.time()
# lsvr = LinearSVR()
lsvc = LinearSVC()
#lsvc = LinearSVC(dual=True,fit_intercept=True,loss='epsilon_insensitive',C=0.8)
lsvc.fit(X_train,Y_train)
# print("--- %s seconds ---" % (time.time() - start_time))


# In[707]:


with open('model/lsvc_c_3.pickle', 'wb') as f:
    pickle.dump(lsvc, f)


# In[615]:


start_time = time.time()
parameters = {'loss':['epsilon_insensitive','squared_epsilon_insensitive'],'C':[1,0.9,1.1]} #C=?
#
#'dual':[True,False],'fit_intercept':[True,False]
clf7 = GridSearchCV(lsvc,parameters,verbose=2,n_jobs=11,scoring='accuracy')
clf7.fit(X_train, Y_train)
print("--- %s seconds ---" % (time.time() - start_time))


# In[616]:


clf7.grid_scores_, clf7.best_params_, clf7.best_score_


# In[691]:


start_time = time.time()
y = lsvc.predict(X_valid)
# y7 = lsvc.predict(X_test)
print("--- %s seconds ---" % (time.time() - start_time))
# y


# In[692]:


ans = []
for i,v in enumerate(y):
    if v < 0.5:
        vv=0
        ans.append(abs(vv-Y_valid[i][0]))
    else:
        vv=1
        ans.append(abs(vv-Y_valid[i][0]))
#     print(i)
acc7= ans.count(0)/len(ans)
print("ACC:",acc7)
#ACC: 0.8512467755803955


# * ridge , kernel ridge ,baseian ridge
# * 踢掉爛feature，創造新feature
#     * ensemble: (model1 ** acc1 * model2 ** acc2) ** (1/(acc1+acc2))，ACC拿trainig predict的ACC，最後在full data 下去train

# In[693]:


## acc_list  to pickle
acc_list=[]
acc3=0
acc_list.append(acc1)
acc_list.append(acc2)
acc_list.append(acc3)
acc_list.append(acc4)
acc_list.append(acc5)
acc_list.append(acc6)
acc_list.append(acc7)
with open('model/c_acc1_7.pickle', 'wb') as f:
    pickle.dump(acc_list, f)


# In[175]:


# 轉道-1,1
y1 = y1*2-1
y2 = y2*2-1
y3 = y3*2-1
y4 = y4*2-1
y5 = y5*2-1
y6 = y6*2-1
y7 = y7*2-1
ens = (y1*acc1+y2*acc2+y3*acc3+y4*acc4+y5*acc5+y6*acc6+y7*acc7)/(acc1+acc2+acc3+acc4+acc5+acc6+acc7) #家群平均
final_ans =[]
for v in ens: #轉回label
    if v < 0:
        final_ans.append(0)
    else:
        final_ans.append(1)
final_ans


# In[176]:


test_data_size = len(X_test)

# # Load parameters
# print('=====Loading Param from %s=====' % save_dir)
# w = np.loadtxt(os.path.join(save_dir, 'w'))
# b = np.loadtxt(os.path.join(save_dir, 'b'))

# # predict
# z = (np.dot(X_test, np.transpose(w)) + b)
# y = sigmoid(z)
# y_ = np.around(y)
Outtput_dir = 'result/'
print('=====Write output to %s =====' % output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_path = os.path.join(output_dir, 'ens_classifier.csv')
with open(output_path, 'w') as f:
    f.write('id,label\n')
    for i, v in  enumerate(final_ans):
        f.write('%d,%d\n' %(i+1, v))


# ## RadiusNeighborsClassifier
# ### 有問題(?

# In[46]:


start_time = time.time()
rnc = RadiusNeighborsClassifier()
rnc.fit(X_train,Y_train)
print("--- %s seconds ---" % (time.time() - start_time))


# In[47]:


start_time = time.time()
y = rnc.predict(X_valid)
print("--- %s seconds ---" % (time.time() - start_time))
y


# In[48]:


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


# ## GPR

# In[165]:


# start_time = time.time()
gpc = GaussianProcessClassifier(n_jobs=11)  #太久，不是合作grid search
# gpr = GaussianProcessRegressor()
gpc.fit(X_train,Y_train)
# print("--- %s seconds ---" % (time.time() - start_time))


# In[181]:


with open('model/gpc_c_1.5.pickle', 'wb') as f:
    pickle.dump(gpc, f)


# In[ ]:


start_time = time.time()
parameters = {'n_restarts_optimizer':[0,1,10],'normalize_y':[True,False],'copy_X_train':[True,False]}
clf3 = GridSearchCV(gpr,parameters,verbose=2,n_jobs=11)
clf3.fit(X_train, Y_train)
print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:


clf3.grid_scores_, clf3.best_params_, clf3.best_score_


# In[166]:


start_time = time.time()
# y = gpc.predict(X_valid)
y3 = gpc.predict(X_test)
print("--- %s seconds ---" % (time.time() - start_time))
# y


# In[142]:


ans = []
for i,v in enumerate(y):
    ans.append(abs(v-Y_valid[i][0]))
#     print(i)
acc3 = ans.count(0)/len(ans)
print("ACC:",acc3)


# ## Select feature

# In[297]:


from xgboost import *
from matplotlib import pyplot
from sklearn.ensemble import *
from functools import reduce


# In[428]:


X_trains = pd.read_csv('data/train_X', sep=',')
X_tests = pd.read_csv('data/test_X',sep=',')
# X_tests


# In[429]:


# X_tests[''
X_trains['divide'] = X_trains['capital_gain'].astype(float)/X_trains['hours_per_week'].astype(float)
X_trains['minus'] = X_trains['capital_gain']-X_trains['capital_loss']
X_tests['divide'] = X_tests['capital_gain'].astype(float)/X_tests['hours_per_week'].astype(float)
X_tests['minus'] = X_tests['capital_gain']-X_tests['capital_loss']
# X_trains
X_trains = np.array(X_trains.values)
X_alls, X_tests = normalize(X_trains, X_tests)
Y_alls = Y_all.ravel()


# In[229]:


# X_trains[X_trains.columns[]]
# X_trains.iloc[0]


# In[245]:


# cols = X_trains.columns
# cols[124]


# In[560]:


# model = XGBClassifier()
# model = ExtraTreesClassifier()
# model = ExtraTreesRegressor()
# model = RandomForestClassifier()
model = RandomForestRegressor()
# model = XGBRegressor()
model.fit(X_alls,Y_alls) #X_all, Y_all


# In[561]:


im_rr = pd.DataFrame()
colses =[]
values = []
for i,v in enumerate(model.feature_importances_):
    if v>0.0014:  #取重要度
        colses.append(cols[i])
        values.append(v)
#         im.append(cols[i],v)
#         print(cols[i],v)
im_rr['column'] = colses
im_rr['important'] = values
im_rr.sort_values(['important'],ascending=False,inplace=True) # 4個地方要改
# im_r #xgb regressor
# im_ #xgb clf
# im_ec #ec clf
# im_er #extratreee regressor
# im_rc #randomforest clf
im_rr # randomforest reg


# In[453]:


im_


# In[436]:


len(im_ec)


# In[562]:


dfc = [im_,im_ec,im_rc]
dfr = [im_r,im_er,im_rr]
dfa = 


# In[584]:


df_merged_c = reduce(lambda  left,right: pd.merge(left,right,on=['column'],
                                            how='inner'), dfc)
df_merged_r = reduce(lambda  left,right: pd.merge(left,right,on=['column'],
                                            how='inner'), dfr)
df_merged_co = reduce(lambda  left,right: pd.merge(left,right,on=['column'],
                                            how='outer'), dfc)
df_merged_ro = reduce(lambda  left,right: pd.merge(left,right,on=['column'],
                                            how='outer'), dfr)


# In[590]:


df_merged_ro


# In[587]:


im_col=[]
for v in df_merged_c['column']: #clf
    im_col.append(v)
print(len(im_col))
#     print(v)
imm_col=[]
for v in df_merged_r['column']: #reg
    imm_col.append(v)
# imrc_col=[]
# for v in im_rc['column']:
#     imrc_col.append(v)
print(len(imm_col))

im_colo=[]
for v in df_merged_co['column']: #clf
    im_colo.append(v)
print(len(im_colo))
#     print(v)
imm_colo=[]
for v in df_merged_ro['column']: #reg
    imm_colo.append(v)
# imrc_col=[]
# for v in im_rc['column']:
#     imrc_col.append(v)
len(imm_colo)


# In[588]:


with open('model/clf_col.pickle', 'wb') as f:
    pickle.dump(im_col, f)
with open('model/reg_col.picke','wb') as f:
    pickle.dump(imm_col,f)
with open('model/clf_colo.pickle', 'wb') as f:
    pickle.dump(im_colo, f)
with open('model/reg_colo.picke','wb') as f:
    pickle.dump(imm_colo,f)


# In[468]:


# im_col


# In[232]:


im = im.sort_values(['important'],ascending=False)
im['column']


# In[250]:


imc = im.merge(imm,how='inner',on="column")
imc


# In[190]:


pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()

