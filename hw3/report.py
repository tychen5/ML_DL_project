
# coding: utf-8

# # Confusion Matrix
# 
# 0.73 model

# In[1]:


from keras.models import load_model
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os, pickle, itertools
from termcolor import colored, cprint
from PIL import Image
from keras import backend as K
from utils import feature, utils
import os, sys
import numpy as np
from random import shuffle
from math import log, floor
import pandas as pd
import tensorflow as tf
import tensorboard as tb
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.activations import *
from keras.callbacks import *
from keras.utils import *
from keras.layers.advanced_activations import *
# from keras.layers.advanced_activations import *
from keras import *
from keras.engine.topology import *
from keras.optimizers import *
import keras
# import pandas as pd
# import numpy as np
# import sklearn
import pickle
from keras.applications import *
from keras.preprocessing.image import *
K.set_image_dim_ordering('tf')

from utils import feature, utils

K.set_learning_phase(0)


# In[2]:


#model_normal
#KUAN_FS_0.73
model = Sequential()
# model.add(Conv2D(64, (3,3), padding='same', input_shape=(48,48,1),kernel_initializer='glorot_normal')) #he_normal
# model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
# model.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
# model.add(BatchNormalization())
# model.add(Dropout(0.25))

model.add(Conv2D(256, (4,4), padding='same',kernel_initializer='glorot_normal', input_shape=(48,48,1))) #same??
# model.add(LeakyReLU(alpha=0.05))
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
# model.add(Dropout(0.25))
model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Dropout(0.25))



# model.add(Conv2D(128, (4,4),kernel_initializer='glorot_normal',padding='same'))
# model.add(LeakyReLU(alpha=0.05))
# model.add(BatchNormalization())
# model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
# model.add(Dropout(0.35))
# model.add(Activation('relu'))

model.add(Conv2D(512, (4,4),kernel_initializer='glorot_normal',padding='same'))
model.add(LeakyReLU(alpha=0.05))
model.add(BatchNormalization())
# model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Dropout(0.35))



model.add(Conv2D(1024, (4,4),kernel_initializer='glorot_normal',padding='same'))
model.add(LeakyReLU(alpha=0.05))  #softplus
# model.add(Activation('softplus'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Dropout(0.4))
# model.add(BatchNormalization())
# model.add(Activation('relu'))

model.add(Conv2D(2048, (3,3),kernel_initializer='glorot_normal',padding='same')) #5,5?
model.add(LeakyReLU(alpha=0.05))  #softplus
# model.add(Activation('softplus'))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
# model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Dropout(0.45))
# model.add(BatchNormalization())

# model.add(Conv2D(256, (3,3),kernel_initializer='glorot_normal',padding='same'))
# # model.add(Activation('softplus'))
# model.add(LeakyReLU(alpha=0.05))
# model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Conv2D(256, (3,3),kernel_initializer='glorot_normal',padding='same'))
# # model.add(Activation('softplus'))
# model.add(LeakyReLU(alpha=0.05))
# model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
# model.add(BatchNormalization())
# # model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))


model.add(Flatten())



# model.add(Dense(2048,kernel_initializer='lecun_normal',kernel_regularizer=regularizers.l2(0.011))) #0.02
# model.add(BatchNormalization())
# model.add(Activation('selu'))
# model.add(Dropout(0.5))

# model.add(Dense(2048,kernel_initializer='lecun_normal',kernel_regularizer=regularizers.l2(0.011))) #0.02=> 512 *2
# model.add(BatchNormalization())
# model.add(Activation('selu'))
# model.add(Dropout(0.5))

# model.add(Dense(512,kernel_initializer='lecun_normal',kernel_regularizer=regularizers.l2(0.001))) #NEW
# model.add(BatchNormalization())
# model.add(Activation('selu'))
# model.add(Dropout(0.5))

# model.add(Dense(512,kernel_initializer='lecun_normal',kernel_regularizer=regularizers.l2(0.001))) #NEW
# model.add(BatchNormalization())
# model.add(Activation('selu'))
# model.add(Dropout(0.5))

model.add(Dense(1024,activation='softplus',kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.02))) #2048,selu,l2
model.add(BatchNormalization())
# model.add(Activation('softplus'))
model.add(Dropout(0.75))

model.add(Dense(1024,kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(0.01))) #2048,selu,l2
model.add(BatchNormalization())
model.add(Activation('softplus'))
model.add(Dropout(0.8))
# model.add(Dense(128,kernel_initializer='glorot_normal',activation='elu',kernel_regularizer=regularizers.l2(0.0001))) #2048,selu,l2
# model.add(BatchNormalization())
# model.add(Activation('softplus'))
# model.add(Dropout(0.5))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(32))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(7))
model.add(Activation('softmax'))
model.summary()
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[112]:


#model_normal
#KUAN_FS_0.7
model = Sequential()
model.add(Conv2D(64, (4,4), padding='same', input_shape=(48,48,1),kernel_initializer='glorot_normal')) #he_normal
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(64, (4,4), padding='same',kernel_initializer='glorot_normal')) #same??
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Dropout(0.25))
model.add(Conv2D(128, (4,4),kernel_initializer='glorot_normal',padding='same'))
model.add(LeakyReLU(alpha=0.05))
model.add(BatchNormalization())
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Dropout(0.35))
model.add(Conv2D(128, (4,4),kernel_initializer='glorot_normal',padding='same'))
model.add(LeakyReLU(alpha=0.05))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Dropout(0.35))
model.add(Conv2D(512, (3,3),kernel_initializer='glorot_normal',padding='same'))
model.add(LeakyReLU(alpha=0.05))  #softplus
model.add(BatchNormalization())
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Dropout(0.4))
model.add(Conv2D(512, (3,3),kernel_initializer='glorot_normal',padding='same')) #5,5?
model.add(LeakyReLU(alpha=0.05))  #softplus
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(512,kernel_initializer='glorot_normal',activation='softplus')) #2048,selu,l2
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7))
model.add(Activation('softmax'))
model.summary()


# In[3]:


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[113]:


def main():
    model_path = os.path.join('model', 'cnn_normal_0.7.h5_weights.h5')
#     emotion_classifier = load_model(model_path)
#     model = load_model(model_path)
    model.load_weights(model_path)
    np.set_printoptions(precision=2)

    with open('data/valid_73.pickle', 'rb') as handle:
        X_valid,Y_valid = pickle.load(handle)
    
#     predictions = emotion_classifier.predict_classes(X_valid)
    ans = model.predict(X_valid)
    predictions=np.argmax(ans,axis=1)
    Y_valid = np.argmax(Y_valid,axis=1)
#     Y_valid = np.array(Y_valid, dtype=np.int32)#.flatten()
    conf_mat = confusion_matrix(Y_valid, predictions)

    plt.figure()
    plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
    fig = plt.gcf()
    dirs= './result/confus'
    fig.savefig(os.path.join(dirs, 'confusion_matrix_07.png'), dpi=600)
    plt.show()


# In[114]:


main()


# #  Saliency Map

# In[11]:


# with open('data/valid_73.pickle', 'rb') as handle:
#     X_valid,Y_valid = pickle.load(handle)
# X_valid = X_valid.reshape((2871,2304))
# private_pixels = [ X_valid[i].reshape((1, 48, 48, 1)) for i in range(len(X_valid)) ]
# print(private_pixels[0].shape)

# img_ids = [i for i in range(5)]
# img_ids


# In[15]:


# private_pixels[3]


# In[15]:


def main2():
    model_name = "cnn_normal_0.7_3.h5_weights.h5"
    model_path = os.path.join('model/', model_name)
    model.load_weights(model_path)
#     model = load_model(model_path)
    print(colored("Loaded model from {}".format(model_name), 'green', attrs=['bold'])) #yellow

    with open('data/valid_73.pickle', 'rb') as handle:
        X_valid,Y_valid = pickle.load(handle)
    X_valid = X_valid.reshape((2871,2304))
    private_pixels = [ X_valid[i].reshape((1, 48, 48, 1)) for i in range(len(X_valid)) ]
#     print(X_valid.shape)
#     private_pixels = [ X_valid.astype('int64')]
    
    input_img = model.input
    img_ids = [i for i in range(5)]

    for idx in img_ids:
        val_proba = model.predict(private_pixels[idx])
        pred = val_proba.argmax(axis=1) #-1
        target = K.mean(model.output[:, pred])
        grads = K.gradients(target, input_img)[0] * 255.0
        fn = K.function([input_img, K.learning_phase()], [grads])
        
#         val_grads = fn([private_pixels[idx].reshape(-1, 48, 48, 1), 0])[0].reshape(48, 48, -1)
#         val_grads *= -1
#         val_grads = np.max(np.abs(val_grads), axis=-1, keepdims=True)
#         heatmap = val_grads.reshape(48, 48)
        
        heatmap = fn([private_pixels[idx]]) #, 0
        heatmap = heatmap[0].reshape(48, 48)
        
        thres = 0.5
        see = private_pixels[idx].reshape(48, 48)
       
        # Plot original image
        plt.figure()
        plt.imshow(see*255.0,cmap='gray')
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(ori_dir, 'original_{}.png'.format(idx)), dpi=600)

        see[np.where(heatmap <= thres)] = np.mean(see)
       
        # Plot saliency heatmap
        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(cmap_dir, 'heatmap_{}.png'.format(idx)), dpi=600)

        # Plot "Little-heat-part masked"
        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(partial_see_dir, '{}.png'.format(idx)), dpi=600)

# if __name__ == "__main__":


# In[16]:


base_dir = os.path.join('./result')
img_dir = os.path.join(base_dir, 'image')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
    os.makedirs(cmap_dir)
partial_see_dir = os.path.join(img_dir,'partial_see')
if not os.path.exists(partial_see_dir):
    os.makedirs(partial_see_dir)
ori_dir = os.path.join(img_dir, 'original')
if not os.path.exists(ori_dir):
    os.makedirs(ori_dir)

main2()


# # Filter

# In[3]:


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)


# In[4]:



def grad_ascent(num_step,input_image_data,iter_func):
   
    for i in range(num_step):
        loss_value, grads_value = iter_func([input_image_data])
        input_image_data += grads_value * 1e-2
    
    filter_images = (input_image_data.reshape(48, 48), loss_value)

    return filter_images


# In[25]:


# from tqdm import *
def main3():
    filter_dir = './result/image'
    store_path = 'filters'
    #model = load_model('./models/cnn_report.h5')
#     model = load_model('./model/cnn_normal_0.7_3.h5')
    model.load_weights('./model/cnn_normal_0.7_3.h5_weights.h5')
    layer_dict = dict([layer.name, layer] for layer in model.layers)
    input_img = model.input

    name_ls = ['conv2d_1']#, 'conv2d_2']
    collect_layers = [ layer_dict[name].output for name in name_ls ]

    NUM_STEPS = 201
    RECORD_FREQ = 50 
    num_step = 201
    nb_filter = 256 #512 OK

    for cnt, c in enumerate(collect_layers):
        filter_imgs = [[] for i in range(NUM_STEPS//RECORD_FREQ)]
        for it in range(NUM_STEPS//RECORD_FREQ):
            for filter_idx in range(nb_filter):
                input_img_data = np.random.random((1, 48, 48, 1)) # random noise
                target = K.mean(c[:, :, :, filter_idx])
                grads = normalize(K.gradients(target, input_img)[0])
                iterate = K.function([input_img], [target, grads])

                filter_imgs[it].append(grad_ascent(num_step, input_img_data, iterate))
        
        for it in range(NUM_STEPS//RECORD_FREQ):
            fig = plt.figure(figsize=(140, 80))
            for i in range(nb_filter):
                ax = fig.add_subplot(nb_filter/16, 16, i+1)
                ax.imshow(filter_imgs[it][i][0], cmap='BuGn')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.xlabel('{:.3f}'.format(filter_imgs[it][i][1]), fontsize=30)
                plt.tight_layout()
            fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[cnt], it*RECORD_FREQ))
            img_path = os.path.join(filter_dir, '{}-{}'.format(store_path, name_ls[cnt]))
            if not os.path.isdir(img_path):
                os.mkdir(img_path)
            fig.savefig(os.path.join(img_path,'e{}'.format(it*RECORD_FREQ)),dpi=100)


# In[26]:


# if __name__ == "__main__":
main3()
# '''
# def main():
#     fig = plt.figure(figsize=(14,8)) # 大小可自行決定
#     for i in range(nb_filter): # 畫出每一個filter
#         ax = fig.add_subplot(nb_filter/16,16,i+1) # 每16個小圖一行
#         ax.imshow(image,cmap='BuGn') # image為某個filter的output或最能activate某個filter的input image
#         plt.xticks(np.array([]))
#         plt.yticks(np.array([]))
#         plt.xlabel('whatever subfigure title you want') # 如果你想在子圖下加小標的話
#         plt.tight_layout()
#     fig.suptitle('Whatever title you want')
#     fig.savefig(os.path.join(img_path,'Whatever filename you want')) #將圖片儲存至disk
# '''


# In[24]:


# 2**16


# In[5]:


# import tqdm
def main32():
    vis_dir = './result/image'
    store_path = 'visual'
    #model = load_model('models/cnn_report.h5')
    #layer_dict = dict([layer.name, layer] for layer in model.layers[1:])
#     model = load_model('model/cnn_normal_0.7_3.h5')
#     model.load_weights()
    model.load_weights('./model/cnn_normal_0.7_3.h5_weights.h5')
    layer_dict = dict([layer.name, layer] for layer in model.layers)

    input_img = model.input
    #name_ls = ["conv2d_2", "conv2d_3"]
    name_ls = ["conv2d_2"]#, "conv2d_2"]
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]

    with open('data/valid_73.pickle', 'rb') as handle:
        x_val,Y_valid = pickle.load(handle)
    
    private_pixels = [ x_val[i].reshape(1, 48, 48, 1) for i in range(len(x_val)) ]

    choose_id = 666
    nb_filter = 512#64
    photo = private_pixels[choose_id]
    c=10
    for cnt, fn in enumerate(collect_layers):
        im = fn([photo]) #get the output of that layer
        fig = plt.figure(figsize=(140, 80))
        nb_filter = im[0].shape[3]
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/16, 16, i+1) #/16,16,i+1
            ax.imshow(im[0][0, :, :, i], cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        fig.suptitle('Output of layer {} (Given image {})'.format(name_ls[cnt], choose_id),fontsize=20)
        img_path = os.path.join(vis_dir, store_path)
        if not os.path.isdir(img_path):
            os.mkdir(img_path)
        fig.savefig(os.path.join(img_path,'layer{}'.format(c)),dpi=100)
        c+=1


# In[6]:


main32()


# 0.7=>VAL0.6819，應可直接load
# 
# 0.7_2=>VAL0.6809，應可直接load
# 
# 0.7_3=>32的非amsgrad版本，應可直接load_model / 0.7_32=>VAL0.6832，只可以load_weights

# In[10]:


store_path='./logs'
train_loss=[]
train_acc=[]
valid_loss=[]
valid_acc=[]
start=2581
end=3186
with open(os.path.join(store_path,'train_loss'),'r') as f:
    for line in f.readlines()[start:end]:
        train_loss.append(float(line))
#     for loss in logs.tr_losses:
#     train_loss = f.read()
#         f.write('{}\n'.format(loss))
with open(os.path.join(store_path,'train_accuracy'),'r') as f:
    for line in f.readlines()[start:end]:
        train_acc.append(float(line))
#     for acc in logs.tr_accs:
#     train_acc = f.read()
#         f.write('{}\n'.format(acc))
with open(os.path.join(store_path,'valid_loss'),'r') as f:
    for line in f.readlines()[start:end]:
        valid_loss.append(float(line))
#     for loss in logs.val_losses:
#     valid_loss = f.read()
#         f.write('{}\n'.format(loss))
with open(os.path.join(store_path,'valid_accuracy'),'r') as f:
    for line in f.readlines()[start:end]:
        valid_acc.append(float(line))
#     for acc in logs.val_accs:
#     valid_acc=f.read()
#         f.write('{}\n'.format(acc))


# In[70]:


import pandas as pd

pd.Series(valid_loss).sort_values(ascending=False).index[0:15]


# max(valid_loss)


# In[67]:


valid_loss[3187]


# In[9]:


import matplotlib.pyplot as plt
def plotDataM(plt,data1,data2):#,data3,data4):
    x1 = [p for p in range(len(data1))]
    y1 = [p for p in data1]
    x2 = [p for p in range(len(data2))]
    y2 = [p for p in data2]
#     x3 = [p[0] for p in data3]
#     y3 = [p[1] for p in data3]
#     x4 = [p[0] for p in data4]
#     y4 = [p[1] for p in data4]
    plt.figure(figsize=(15,10))
    axes = plt.gca()
    axes.set_ylim([0,45]) #min max1
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x1, y1,color='b') #train 藍色
    plt.plot(x2, y2,color='r') #valid 紅色
#     plt.plot(x3, y3,color='b')  #0.1
#     plt.plot(x4, y4, '-o',color='y')  #10


# In[11]:


plotDataM(plt,train_loss,valid_loss)
fig = plt.figure(figsize=(140, 80))
fig.savefig('result/accuracy.jpg',dpi=100)
plt.show()


# In[50]:


# train_losses = train_loss.split('\n')
# train_loss = [float(i) for i in train_losses]
# type(train_loss[0])


# In[43]:


# for i in train_loss:
#     print(i)


# In[ ]:


# import os
# import argparse
# from keras.models import load_model
# from termcolor import colored
# from termcolor import cprint
# import keras.backend as K
# from utils import *
# import numpy as np
# import matplotlib.pyplot as plt

# base_dir = './'
# img_dir = os.path.join(base_dir, 'image')
# if not os.path.exists(img_dir):
#     os.makedirs(img_dir)
# cmap_dir = os.path.join(img_dir, 'cmap')
# if not os.path.exists(cmap_dir):
#     os.makedirs(cmap_dir)
# partial_see_dir = os.path.join(img_dir,'partial_see')
# if not os.path.exists(partial_see_dir):
#     os.makedirs(partial_see_dir)
# origin_dir = os.path.join(img_dir,'origin')
# if not os.path.exists(origin_dir):
#     os.makedirs(origin_dir)

# def read_data(filename, label=True, width=48, height=48):
#     width = height = 48
#     with open(filename, 'r') as f:
#         data = f.read().strip('\r\n').replace(',', ' ').split()[2:]
#         data = np.array(data)
#         _X = np.delete(data, range(0, len(data), width*height+1), axis=0).reshape((-1, width, height, 1)).astype('int')
#         Y = data[::width*height+1].astype('int')

#         X = _X

#         X = X.astype('float') / 255

#         if label:
#             return X, Y, _X
#         else:
#             return X

# def main():
#     parser = argparse.ArgumentParser(prog='saliency_map.py',
#             description='ML-Assignment3 visualize attention heat map.')
#     parser.add_argument('--model', type=str, metavar='<#model>', required=True) #load_model
#     parser.add_argument('--data', type=str, metavar='<#data>', required=True)
#     parser.add_argument('--attr', type=str, metavar='<#attr>', required=True) #mean,std
#     args = parser.parse_args()
#     data_name = args.data
#     model_name = args.model
#     attr_name = args.attr

#     attr = np.load(attr_name)
#     mean, std = attr[0], attr[1]

#     emotion_classifier = load_model(model_name)

#     print(colored("Loaded model from {}".format(model_name), 'yellow', attrs=['bold']))

#     X, Y, _X = read_data(data_name, label=True)
#     X = (X - mean) / (std + 1e-20)

#     input_img = emotion_classifier.input
#     img_ids = [-4998]

#     for idx in img_ids:
#         val_proba = emotion_classifier.predict(X[idx].reshape(-1, 48, 48, 1))
#         pred = val_proba.argmax(axis=-1)
#         target = K.mean(emotion_classifier.output[:, pred])
#         grads = K.gradients(target, input_img)[0]
#         fn = K.function([input_img, K.learning_phase()], [grads])

#         val_grads = fn([X[idx].reshape(-1, 48, 48, 1), 0])[0].reshape(48, 48, -1)
       
#         val_grads *= -1
#         val_grads = np.max(np.abs(val_grads), axis=-1, keepdims=True)

#         # normalize
#         val_grads = (val_grads - np.mean(val_grads)) / (np.std(val_grads) + 1e-5)
#         val_grads *= 0.1

#         # clip to [0, 1]
#         val_grads += 0.5
#         val_grads = np.clip(val_grads, 0, 1)

#         # scale to [0, 1]
#         val_grads /= np.max(val_grads)

#         heatmap = val_grads.reshape(48, 48)

#         print('ID: {}, Truth: {}, Prediction: {}'.format(idx, Y[idx], pred))
#         # show original image
#         plt.figure()
#         plt.imshow(_X[idx].reshape(48, 48), cmap='gray')
#         plt.colorbar()
#         plt.tight_layout()
#         fig = plt.gcf()
#         plt.draw()
#         fig.savefig(os.path.join(origin_dir, '{}.png'.format(idx)), dpi=100)

#         thres = 0.55
#         see = _X[idx].reshape(48, 48)
#         see[np.where(heatmap <= thres)] = np.mean(see)

#         plt.figure()
#         plt.imshow(heatmap, cmap=plt.cm.jet)
#         plt.colorbar()
#         plt.tight_layout()
#         fig = plt.gcf()
#         plt.draw()
#         fig.savefig(os.path.join(cmap_dir, '{}.png'.format(idx)), dpi=100)

#         plt.figure()
#         plt.imshow(see,cmap='gray')
#         plt.colorbar()
#         plt.tight_layout()
#         fig = plt.gcf()
#         plt.draw()
#         fig.savefig(os.path.join(partial_see_dir, '{}.png'.format(idx)), dpi=100)

# if __name__ == "__main__":
#     main()

