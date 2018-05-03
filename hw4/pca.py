
# coding: utf-8

# # Reconstruct pca.py

# In[113]:

try:
    import numpy as np
    from skimage import io
    from skimage.io import *
    import skimage
    from os import listdir
    import os
    import sys
    from skimage import transform
except:
    pass
# In[98]:


# num_PCA = 4
image_T = sys.argv[2]#'data/Aberdeen/13.jpg'
imgDir = sys.argv[1]
imgName = listdir(imgDir)
image_X=[]
for name in imgName:
    img_one = io.imread(os.path.join(imgDir,name))
    image_X.append(img_one.flatten())


# In[99]:


train_X = np.array(image_X)
print(train_X.shape)


# In[100]:


# train_X = train_X.astype('float64')
mean_face = np.mean(train_X,axis=0)
img_cent = train_X - mean_face
# train_X -= mean_face


# In[101]:


U,s,_ = np.linalg.svd(img_cent.T, full_matrices=False)
print(U.shape)


# In[102]:


test_X = io.imread(os.path.join(imgDir,image_T)).flatten()
test_X_cent = test_X - mean_face


# In[103]:


weight = np.dot(test_X_cent,U[:,:4])
print(weight.shape)


# In[104]:


# for i,name in enumerate(imgName):
#     if name.endswith(image_T.split('/')[-1]):
#         img_ind = i


# In[105]:


# rec_face= weight[img_ind,:4].dot(U.T[:4])+mean_face #4
rec_face = np.dot(weight,U[:,:4].T)+mean_face
# rec_face.resize(600,600,3)
print(rec_face.shape)


# In[106]:


# rec_face[rec_face<0]=0
rec_face -= np.min(rec_face)
rec_face /= np.max(rec_face)
rec_face = (rec_face*255).astype(np.uint8)
# new_img = transform.resize(rec_face, (600,600,3))
rec_face.resize(600,600,3) #inplace=True


# In[107]:


imsave('reconstruction.png',rec_face) #change
