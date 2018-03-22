
# coding: utf-8

# In[157]:


import sys
import csv 
import math
import random
import numpy as np


# In[158]:


data = []    
#一個維度儲存一種污染物的資訊
for i in range(18):
	data.append([])

n_row = 0
text = open('data/train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列為header沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))	
    n_row = n_row+1
text.close()


# In[159]:


x = []
y = []

for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        x.append([])
        # 總共有18種污染物
        for t in range(18):
            # 連續9小時
            for s in range(9):
                x[471*i+j].append(data[t][480*i+j+s] )
        y.append(data[9][480*i+j+9])
x = np.array(x)
y = np.array(y)

# x = np.concatenate((x,x**2), axis=1)
# 增加平方項

x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
# 增加bias項                   


# In[80]:


x.shape


# In[164]:


w = np.zeros(len(x[0]))         # initial weight vector
lr = 1    #1e-6, 0.01 , 0.1,1 (22.8) ,10(22.9),100(22.9)            # learning rate
iters = 1000    
lambda_ = 0.99 #1e-6, 0.001, 0.05, 0.99  #越大越強


# In[168]:


x_t = x.transpose()
s_gra = np.zeros(len(x[0]))
plotsL4 = []
for i in range(iters):
    hypo = np.dot(x,w)
    loss = hypo - y
    cost = np.sum(loss**2) / len(x)
    cost_a  = math.sqrt(cost) + lambda_ * np.linalg.norm(w)/len(x)
#     cost_a  = math.sqrt(cost)
    gra = np.dot(x_t,loss)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
#     w = w - lr * gra/ada
    w = w - lr * (gra/ada + 2*lambda_*w)
    print ('iteration: %d | Cost: %f  ' % ( i,cost_a))
    if i > 1 :
        plotsL4.append((i,cost_a))


# In[124]:


import matplotlib.pyplot as plt
def plotData(plt,data):
    x = [p[0] for p in data]
    y = [p[1] for p in data]
#     plt.figure(figsize=(15,10))
    axes = plt.gca()
    axes.set_ylim([20,50]) #min max
    plt.xlabel("epoch")
    plt.ylabel("cost")
    plt.plot(x, y,color='r')


# In[75]:


import matplotlib.pyplot as plt
def plotDataM(plt,data1,data2,data3,data4):
    x1 = [p[0] for p in data1]
    y1 = [p[1] for p in data1]
    x2 = [p[0] for p in data2]
    y2 = [p[1] for p in data2]
    x3 = [p[0] for p in data3]
    y3 = [p[1] for p in data3]
    x4 = [p[0] for p in data4]
    y4 = [p[1] for p in data4]
    plt.figure(figsize=(15,10))
    axes = plt.gca()
    axes.set_ylim([20,50]) #min max
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.plot(x1, y1, '-o',color='r') #1e-6
    plt.plot(x2, y2, '-o',color='g') #0.01
    plt.plot(x3, y3, '-o',color='b')  #0.1
    plt.plot(x4, y4, '-o',color='y')  #10


# In[171]:


# plotData(plt,plotsL1)
plotDataM(plt,plotsL1,plotsL2,plotsL3,plotsL4)
plt.show()


# # TEST

# In[169]:


test_x = []
n_row = 0
text = open('data/test.csv' ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row %18 == 0:
        test_x.append([])
        for i in range(2,11):
            test_x[n_row//18].append(float(r[i]) )
    else :
        for i in range(2,11):
            if r[i] !="NR":
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()
test_x = np.array(test_x)

# test_x = np.concatenate((test_x,test_x**2), axis=1)
# 增加平方項

test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)
# 增加bias項  


# In[170]:


ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

filename = 'result/L4.csv'
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()

