
# coding: utf-8

# In[3]:


import tensorflow as tf
import numpy as np
import pandas as pd


# In[2]:


def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid


# In[7]:


img = np.load('data/image.npy')
img = img/255
img.shape


# In[21]:


num_epoch = 100
batch_size =32
path='./model/auto.ckpt'


# In[14]:


#入口
X = tf.placeholder('float', [None,784],name='input_X') #input shape，入口形狀。輸入資料的資料型態, list, 名字。等同Y

# Encoder
hidden = tf.layers.dense(inputs=X ,units=512, activation=tf.nn.relu ) #layer
hidden =  tf.layers.dense(inputs=hidden ,units=256, activation=tf.nn.relu )

code =  tf.layers.dense(inputs=hidden ,units=128, activation=tf.nn.relu )

# Decoder
hidden =  tf.layers.dense(inputs=code ,units=256, activation=tf.nn.relu )
logits = tf.layers.dense(inputs=hidden,units=784, activation=tf.nn.relu )


# In[20]:


# loss_op = tf.reduce_mean()
loss_op = tf.losses.mean_squared_error(labels=X, predictions=logits) #imput,nn output: labels
optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
train_op = optimizer.minimize(loss_op)
saver = tf.train.Saver()
init = tf.global_variables_initializer() #初始化


# In[26]:


with tf.Session() as sess: #是一個model，是一個GPU，是一個引擎，一個可跑的
    sess.run(init) #真正初始化
    for epoch in range(1,num_epoch + 1):
        choice = np.random.choice(img.shape[0], img.shape[0], replace=False) #不要重複取，隨機index，拿那些index
        num_batch = int(img.shape[0]/batch_size)+1
        batches = np.array_split(choice, num_batch)
        for batch in batches:
            X_batch = img[batch] #每次要拿哪些index近來tain
            sess.run(train_op,feed_dict={X:X_batch}) # X的變樹名稱對應的data，Y變數名稱(place holder)對應的data
        loss = sess.run(loss_op,feed_dict={X:img}) #看誰的loss，這邊是全部X_train，可以有多個loss
        print(epoch,'train_loss',loss)
        
    saver.save(sess=sess,save_path=path) #存起來了


# ## predict

# In[28]:


with tf.Session() as sess:
    saver.restore(sess,path)
    feature = sess.run(code,feed_dict={X:img}) #會把每張圖片的vector存進去feature list裡面
    
    np.save('data/feature.npy',feature)


# ***

# In[1]:


from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import *
from keras import backend as K
from keras.callbacks import *
from keras.layers import *
from keras.optimizers import *
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.decomposition import *
from sklearn.cluster import *


# In[2]:


img = np.load('data/image.npy')
img = img/255
# train_X = img.reshape(-1,28,28,1)
# train_X.shape, img.shape


# In[84]:




input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(512, (5, 5), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(256, (4, 4), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
encoder = Model(input_img,encoded)
# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x2 = Flatten()(encoded)
x2 = Dense(128,activation='relu')(x2)
x2 = Dense(784)(x2)
# decoded = K.reshape(x2,(28,28,1))
decoded = Reshape((28,28,1))(x2)

# # x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
# # x = UpSampling2D((2, 2))(x)
# x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(128, (3, 3), activation='relu')(x)
# x = UpSampling2D((2, 2))(x)
# # x = Conv2D(512, (3, 3), activation='relu')(x)
# # x = UpSampling2D((2, 2))(x)
# decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adamax', loss='mse', metrics=['accuracy']) #binary_crossentropy
autoencoder.summary()
# encoder.summary()


# In[3]:


#DNN autoencoder
input_img = Input(shape=(784,))
x = Dense(512,activation='relu')(input_img) #one_norm #activity
x = Dense(512,activation='relu')(x)
x = Dense(512,activation='relu')(x)
x = Dense(256,activation='relu')(x)
x = Dense(256,activation='relu')(x)
x = Dense(256,activation='relu')(x)
x = Dense(128,activation='relu')(x) #code # 改成256?
encoder = Model(inputs=input_img,outputs=x)
d = Dense(256)(x)
d = Dense(512)(d)
d = Dense(784,activation='sigmoid')(d)
autoencoder = Model(inputs=input_img,outputs=d)
opt = Nadam()
autoencoder.compile(optimizer=opt,loss='mse')
autoencoder.summary()


# In[37]:


# np.mean(img)


# In[145]:


img = np.load('data/image.npy')
img = img/255
# img = (img - np.mean(img))/np.std(img)
batchSize=128
patien=15
epoch=500
saveP = 'model/autoencoder_keras_DNN2.h5'
logD = './logs/'
callback=[
    EarlyStopping(patience=patien,monitor='val_loss',verbose=1),
    ModelCheckpoint(saveP,monitor='val_loss',verbose=1,save_best_only=True, save_weights_only=True),
]
autoencoder.fit(img, img,
                epochs=epoch,
                batch_size=batchSize,
                shuffle=True,
                validation_split=0.1,
                callbacks=callback, 
#                 class_weight='auto'
                )
autoencoder.save(saveP+"_all.h5")
encoder.save(saveP+'_enc_all.h5')
encoder.save_weights(saveP+'_enc.h5')
#0.152X


# In[4]:


encoder.load_weights('model/autoencoder_keras_DNN2.h5_enc.h5')
feature = encoder.predict(img)
features = feature #DNN
len(feature[0].flatten())


# In[154]:


np.save('data/feature_keras_DNN2.npy',feature)


# In[87]:


features = feature.reshape(140000,512)


# In[9]:


# #prepro
# img_path = 'data/image.npy'
test_path = 'data/test_case.csv'
predict_path='./result/output2_auto_DNN2_test.csv'
# img = np.load(img_path)
# img = img/255
# train_X = img/255
test_X = pd.read_csv(test_path)
# if not os.path.exists(os.path.join(*predict_path.split('/')[:-1])):
#     os.makedirs(os.path.join(*predict_path.split('/')[:-1]))


# In[6]:



cluster = KMeans(init='k-means++',n_init=35,max_iter=350,precompute_distances='auto',algorithm='auto',random_state=0
                 ,n_clusters=2,n_jobs=11,verbose=0) #11,305 k-means++
cluster.fit(features)


# In[1]:


sum(cluster.labels_)
#55883<69107 <70000 <70893<75178<76434 <99175


# In[10]:


all_ind = test_X['ID']
id1= test_X['image1_index']
id2= test_X['image2_index']

with open(predict_path, 'w') as f:
    f.write('ID,Ans\n')
    for i in range(len(all_ind)):
        if cluster.labels_[id1[i]] == cluster.labels_[id2[i]]:
            f.write('%d,1\n'%all_ind[i])
        else:
            f.write('%d,0\n'%all_ind[i])

