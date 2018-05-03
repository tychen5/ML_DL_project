
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.decomposition import *
from sklearn.cluster import KMeans
import sys, os
import pickle
import tqdm ##
from tqdm import * ##


# In[13]:


# predict_path.split('/')[:-1]


# In[47]:


#prepro
img_path = sys.argv[1]#'data/image.npy'
test_path = sys.argv[2]#'data/test_case.csv'
predict_path= sys.argv[3]#'./result/train_method1.csv'
img = np.load(img_path)
train_X = img/255
test_X = pd.read_csv(test_path)
# if not os.path.exists(os.path.join(*predict_path.split('/')[:-1])):
#     os.makedirs(os.path.join(*predict_path.split('/')[:-1]))


# # Method 2

# In[49]:



pca = PCA(n_components=415,iterated_power='auto', whiten=True,svd_solver="full",random_state=725035) #n_components='mle',395
train_X_PCA = pca.fit_transform(train_X)

print("======PCA DONE========")
cluster = KMeans(init='k-means++',n_init=10,max_iter=350,precompute_distances='auto',algorithm='auto',random_state=725035
                 ,n_clusters=2,n_jobs=-1,verbose=0) #11,305
cluster.fit(train_X_PCA)

pickle.dump(pca,open('models/method2_pca.pkl','wb'))
pickle.dump(cluster, open('models/method2_mle2.pkl','wb'))


# In[32]:


#predict
cluster = pickle.load(open('models/method2_mle2.pkl','rb'))


all_ind = test_X['ID']
id1= test_X['image1_index']
id2= test_X['image2_index']
print(sum(cluster.labels_)) #=70000
with open(predict_path, 'w') as f:
    f.write('ID,Ans\n')
    for i in range(len(all_ind)):
        if cluster.labels_[id1[i]] == cluster.labels_[id2[i]]:
            f.write('%d,1\n'%all_ind[i])
        else:
            f.write('%d,0\n'%all_ind[i])


# In[50]:


# pickle.dump(pca,open('model/method2_pca.pkl','wb'))


# ***
# *test*

# # method 1

# In[4]:


# train_X = np.load('data/image_old.npy')
# train_X = train_X / 255 #轉換到0~1

# TRAIN = False

# if TRAIN:
min_ = 140000
for i in tqdm(range(50)):
    pca = PCA(n_components=280, whiten=True, svd_solver='randomized').fit_transform(train_X)

    kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10,n_jobs=11)
    kmeans.fit(pca)

    sum_ = sum(kmeans.labels_)

    if abs(sum_-70000) <= min_:
        min_ = abs(sum_-70000) 
        label = kmeans.labels_
        print(i,sum_, label)
        if sum_ == 70000:
            break

pickle.dump(label, open('model/method1.pkl','wb'))

# else:


# In[48]:


label = pickle.load(open('model/method1.pkl','rb'))

# test_X = pd.read_csv('data/test_case_old.csv') #old
ii = test_X['ID']
id1= test_X['image1_index']
id2= test_X['image2_index']

with open(predict_path, 'w') as f:
    f.write('ID,Ans\n')
    for i in range(len(ii)):
        if label[id1[i]] == label[id2[i]]:
            f.write('%d,1\n'%ii[i])
        else:
            f.write('%d,0\n'%ii[i])


# ## Others
# * sklearn.decomposition.TruncatedSVD
# * sklearn.decomposition.NMF
# * sklearn.decomposition.PCA
# 
# *others*
# 
# - sklearn.lda
# - sklesrn.featureselection

# 1）對數據先fit，再transform，好處是我可以拿到數據變換(比如scaling/幅度變換/標準化)的參數，這樣你可以在測試集上也一樣做相同的數據變換處理
# 
# 2）fit_trainsform，一次性完成數據的變換(比如scaling/幅度變換/標準化)，比較快。但是如果在訓練集和測試集上用fit_trainsform，可能執行的是兩套變換標準(因為訓練集和測試集幅度不一樣) 
# 

# In[36]:


mf = NMF()
train_X_MF = mf.fit_transform(train_X)

clusterM = KMeans(init='k-means++',n_init=10,max_iter=350,precompute_distances='auto',algorithm='auto',random_state=725035
                 ,n_clusters=2,n_jobs=11,verbose=0) #11,305
clusterM.fit(train_X_MF)


# In[43]:


# sum(clusterM.labels_)
all_ind = test_X['ID']
id1= test_X['image1_index']
id2= test_X['image2_index']
print(sum(clusterM.labels_)) #=70000
with open(predict_path, 'w') as f:
    f.write('ID,Ans\n')
    for i in range(len(all_ind)):
        if clusterM.labels_[id1[i]] == clusterM.labels_[id2[i]]:
            f.write('%d,1\n'%all_ind[i])
        else:
            f.write('%d,0\n'%all_ind[i])


# In[ ]:


2.25e-1

pca.explained_variance_ratio_

