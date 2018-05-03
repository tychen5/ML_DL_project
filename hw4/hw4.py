
# coding: utf-8

# In[1]:

try:
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import *
    from sklearn.cluster import KMeans
    import sys, os
    import pickle
    import tensorflow
    import keras
except:
    pass

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



#pca = PCA(n_components=415,iterated_power='auto', whiten=True,svd_solver="full",random_state=725035) #n_components='mle',395
#train_X_PCA = pca.fit_transform(train_X)

print("======PCA DONE========")
#cluster = KMeans(init='k-means++',n_init=10,max_iter=350,precompute_distances='auto',algorithm='auto',random_state=725035,n_clusters=2,n_jobs=-1,verbose=0) #11,305
#cluster.fit(train_X_PCA)

#pickle.dump(pca,open('models/method2_pca.pkl','wb'))
#pickle.dump(cluster, open('models/method2_mle2.pkl','wb'))


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




