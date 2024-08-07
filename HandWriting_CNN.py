# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 20:32:32 2021

@author: zhaodf
"""
#采用CNN对MNIST手写数字进行识别
#MNIST是一个手写体数字的图片数据集，该数据集来由美国国家标准与技术研究所
# （National Institute of Standards and Technology (NIST)）发起整理
#一共统计了来自250个不同的人手写数字图片，其中50%是高中生，50%来自人口普查局的工作人员
#在上述数据集中，训练集一共包含了 60,000 张图像和标签，而测试集一共包含了 10,000 张图像和标签

###################数据加载####################################
from keras.datasets import mnist
import matplotlib.pyplot as plt
(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.subplot(241)
plt.imshow(X_train[12], cmap=plt.get_cmap('gray'))
plt.subplot(242)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(243)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(244)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
plt.subplot(245)
plt.imshow(X_train[4], cmap=plt.get_cmap('gray'))
plt.subplot(246)
plt.imshow(X_train[5], cmap=plt.get_cmap('gray'))
plt.subplot(247)
plt.imshow(X_train[6], cmap=plt.get_cmap('gray'))
plt.subplot(248)
plt.imshow(X_train[7], cmap=plt.get_cmap('gray'))

plt.show()


import matplotlib.pyplot as plt
import scipy.io as scio
from keras.utils import np_utils # 导入np_utils是为了用one hot encoding方法将输出标签的向量（vector）转化为只在出现对应标签的那一列为1，其余为0的布尔矩阵
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers import Flatten, Dense
from keras.layers import MaxPooling2D, Dropout
from keras.layers import Input, Activation, Conv2D

(X_train,y_train),(X_test,y_test) = mnist.load_data() #加载数据

#给定的像素的灰度值在0-255，为了使模型的训练效果更好，通常将数值归一化映射到0-1
X_train = X_train / 255
X_test = X_test / 255

X_train=X_train.reshape(60000,28,28,1)
X_test=X_test.reshape(10000,28,28,1)

# one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# 搭建神CNN网络模型 ，创建一个函数，建立含有一个隐层的神经网络
def HW_CNN(input_shape):
    
    X_input = Input(input_shape)
    
    X = Conv2D(32, kernel_size=(2,2), padding='same', strides=(2,2),name = 'layer1')(X_input)    
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2), name='max_pool1')(X)
     
    X = Conv2D(32, kernel_size=(2,2), padding='same', strides=(2,2),name = 'layer2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2), name='max_pool2')(X)

    X = Flatten()(X)

    X = Dense(64, name='fc1')(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X) 
    X = Dense(10, activation='softmax')(X)

    HW_CNN = Model(inputs=X_input,output=X)
    
    return HW_CNN
##############################网络训练并保存#######################################################
HW_CNN = HW_CNN(input_shape=(28,28,1))
HW_CNN.summary()#模型参数结构
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)#优化器
HW_CNN.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ["accuracy"])#交叉熵

from time import time
start = time()
history=HW_CNN.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)#训练
end = time()
print("CPU_time =" + str(end-start))

HW_CNN.save('HW_CNN.h5')#保存模型

y1=history.history['acc']#保存模型训练情况
y2=history.history['val_acc']
y3=history.history['loss']
y4=history.history['val_loss']
scio.savemat('acc.mat', {'data': y1 }) 
scio.savemat('val_acc.mat', {'data': y2 })
scio.savemat('loss.mat', {'data': y3 })
scio.savemat('val_loss.mat', {'data': y4 }) ##示例：存为mat文件
##############################绘制网络训练曲线#####################
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}

#fig = plt.figure()
figsize = 7,5
figure, ax = plt.subplots(figsize=figsize)
plt.tick_params(labelsize=12)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.plot(history.history['acc'],'-r',label='training acc',linewidth=1.5)
plt.plot(history.history['val_acc'],'-b',label='val acc',linewidth=1.5)
plt.title('model accuracy',font2)
plt.ylabel('accuracy',font2)
plt.xlabel('epoch',font2)
plt.legend(loc='lower right',prop=font2)

#fig = plt.figure()
figsize = 7,5
figure, ax = plt.subplots(figsize=figsize)
plt.tick_params(labelsize=12)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.plot(history.history['loss'],'-r',label='training loss',linewidth=1.5)
plt.plot(history.history['val_loss'],'-b', label='val loss',linewidth=1.5)
plt.title('model loss',font2)
plt.ylabel('loss',font2)
plt.xlabel('epoch',font2)
plt.legend(loc='upper right',prop=font2)

figsize = 7,5
figure, ax = plt.subplots(figsize=figsize)
plt.tick_params(labelsize=12)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.plot(history.history['acc'],'-g',label='training acc',linewidth=1.5)
plt.plot(history.history['val_acc'],'-r',label='val acc',linewidth=1.5)
plt.plot(history.history['loss'],'-y',label='training loss',linewidth=1.5)
plt.plot(history.history['val_loss'],'-b', label='val loss',linewidth=1.5)
plt.title('model loss and accuracy',font2)
plt.ylabel('value',font2)
plt.xlabel('epoch',font2)
plt.legend(loc='best',prop=font2)

##################模型评估##################################
d1_model = load_model('HW_CNN.h5')
start2 = time()
preds = d1_model.evaluate(x = X_test, y = y_test)
end2 = time()
print("CPU_time2 =" + str(end2-start2))
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))#打印精度
