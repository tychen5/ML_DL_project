
# coding: utf-8

# In[29]:


import os, sys
import numpy as np
from random import shuffle
#import argparse
from math import log, floor
import pandas as pd


# In[32]:


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


# In[87]:


def load_data(train_data_path, train_label_path, test_data_path):
    X_train = pd.read_csv(train_data_path, sep=',', header=0)
#     X_train = X_train.drop(columns='age')
    X_train = np.array(X_train.values)
    Y_train = pd.read_csv(train_label_path, sep=',', header=0)
    Y_train = np.array(Y_train.values)
    X_test = pd.read_csv(test_data_path, sep=',', header=0)
#     X_test = X_test.drop(columns='age')
    X_test = np.array(X_test.values)
#     print(X_train.shape,Y_train.shape)
    return (X_train, Y_train, X_test)


# In[7]:


def valid(w, b, X_valid, Y_valid):
    valid_data_size = len(X_valid)

    z = (np.dot(X_valid, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    print('Validation acc = %f' % (float(result.sum()) / valid_data_size))
    return


# In[33]:


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


# ## Predict Testing data

# In[109]:


def infer(X_test, save_dir, output_dir):
    test_data_size = len(X_test)

    # Load parameters
#     print('=====Loading Param from %s=====' % save_dir)
    w = np.loadtxt(os.path.join(save_dir, 'w'))
    b = np.loadtxt(os.path.join(save_dir, 'b'))

    # predict
    z = (np.dot(X_test, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)

    print('Write  to %s ' % output_dir)
#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)
    output_path = output_dir
    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_):
            f.write('%d,%d\n' %(i+1, v))

    return


# In[49]:


def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
#     print(X.shape, Y.shape)
    return (X[randomize], Y[randomize])


# In[107]:


def train(X_all, Y_all, save_dir):
    # 50%
    valid_set_percentage = 0.5
    X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)
    print("X_train,X_valid:",len(X_train),len(X_valid))
    dim=123
    #  hyperparameter
    w = np.zeros((dim,))  #123
    b = np.zeros((1,))
    l_rate = 0.22
    eps = 1e-8
    decay = 0.995
    batch_size = 32
    train_data_size = len(X_train)
    step_num = int(floor(train_data_size / batch_size))
    epoch_num = 4122
    save_param_iter = 229
    lambda_ = 0.1 #0,1e-3,0.1

    # Start training
    total_loss = 0.0
    for epoch in range(1, epoch_num):
        # Do validation and parameter saving
        if (epoch) % save_param_iter == 0:
            print('=====Saving Param at epoch %d=====' % epoch)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            np.savetxt(os.path.join(save_dir, 'w_report'), w)
            np.savetxt(os.path.join(save_dir, 'b_report'), [b,])
            print('epoch avg loss = %f' % (total_loss / (float(save_param_iter) * train_data_size)))
            total_loss = 0.0
            valid(w, b, X_valid, Y_valid)

        X_train, Y_train = _shuffle(X_train, Y_train)

        # Train on batch
        for idx in range(step_num):
            X = X_train[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]

            z = np.dot(X, np.transpose(w)) + b
            y = sigmoid(z)

            cross_entropy = -1 * (np.dot(np.squeeze(Y), np.log(y)) + np.dot((1 - np.squeeze(Y)), np.log(1 - y))) + 0.5*lambda_ * np.linalg.norm(w)
            total_loss += cross_entropy 

            w_grad = np.mean(-1 * X * (np.squeeze(Y) - y).reshape((batch_size,1)), axis=0)
            b_grad = np.mean(-1 * (np.squeeze(Y) - y))

            
            w = w - l_rate * (w_grad + lambda_*w)
            b = b - l_rate * b_grad
        if l_rate > eps: #loss一樣的話就加大lr
            l_rate*=decay

    return


# ## Params

# In[110]:


def main(test_x_path,predict_path,train_x_path,train_y_path):
    # Load feature and label
    X_all, Y_all, X_test = load_data(train_x_path,train_y_path, test_x_path)
    # Normalization
    X_all, X_test = normalize(X_all, X_test)
    
    trains = False
    infers = True
    # To train or to infer
    if trains:
        train(X_all, Y_all, 'model_old/')
    if infers:
        infer(X_test,  'model_old/', predict_path)
    return

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])


# In[35]:


# X_Train


# In[69]:


# X_train = pd.read_csv('data/train_X', sep=',', header=0)
# X_train

