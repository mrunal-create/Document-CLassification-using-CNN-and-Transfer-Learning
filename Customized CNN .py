#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow')


# In[4]:


pip install opencv-python


# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import string
import nltk
import pathlib
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, BinaryAccuracy, Precision, Recall, AUC


# In[2]:


datasetFolder = "C:/Users/hp/OneDrive/Desktop/SML Project/datasets/train"
datasetFolder1 = "C:/Users/hp/OneDrive/Desktop/SML Project/datasets/val"


# In[3]:


train = pathlib.Path(os.path.join(datasetFolder))
val=pathlib.Path(os.path.join(datasetFolder1))


# In[9]:


def label_images_preprocess(images, label):
  arr = []
  labels = []
  for i in images:
    img = cv2.imread(os.path.join(i))
    plt.figure(figsize = (10, 10))
    plt.imshow(img)
    plt.grid(False)
    plt.show()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize = (10, 10))
    plt.imshow(img)
    plt.grid(False)
    plt.show()
    img_mean = np.mean(img)
    img = img - img_mean
    img = img / np.std(img)
    plt.figure(figsize = (10, 10))
    plt.imshow(img)
    plt.grid(False)
    plt.show()
    img = cv2.GaussianBlur(img,(5,5),0)
    plt.figure(figsize = (10, 10))
    plt.imshow(img)
    plt.grid(False)
    plt.show()
    #optimalThreshold,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    img = cv2.resize(img, (100, 100))
    arr.append(img)
    labels.append(label)
  return [arr, labels]


# In[10]:


[form, Y_form] = label_images_preprocess(list(val.glob("form/*.*")), 0)
[invoice, Y_invoice] = label_images_preprocess(list(val.glob("invoice/*.*")), 1)
[letter, Y_letter] = label_images_preprocess(list(val.glob("letter/*.*")), 2)
[resume, Y_resume] = label_images_preprocess(list(val.glob("resume/*.*")), 3)


# In[41]:


test_images = form + invoice + letter +resume
test_labels =  Y_form +Y_invoice+ Y_letter +Y_resume 

test_images = np.asarray(test_images)
test_labels = np.asarray(test_labels)


# In[7]:


[form, Y_form] = label_images_preprocess(list(train.glob("form/*.*")), 0)
[invoice, Y_invoice] = label_images_preprocess(list(train.glob("invoice/*.*")), 1)
[letter, Y_letter] = label_images_preprocess(list(train.glob("letter/*.*")), 2)
[resume, Y_resume] = label_images_preprocess(list(train.glob("resume/*.*")), 3)


# In[36]:


train_images = form + invoice + letter +resume
train_labels =  Y_form +Y_invoice+ Y_letter +Y_resume 


# In[37]:


import random
temp = list(zip(train_images, train_labels))
random.shuffle(temp)
train_images, train_labels = zip(*temp)
train_images, train_labels = list(train_images), list(train_labels)


# In[38]:


train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)


# In[43]:


train_labels = to_categorical(train_labels)
test_labels=to_categorical(test_labels)


# In[44]:


for i in range(0,5):
    plt.figure(figsize = (1, 1))
    plt.imshow(train_images[i])
    plt.grid(False)
    plt.show()


train_images.shape


# In[28]:


m = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=70, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(100,100,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=70, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=60, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=60, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
   tf.keras.layers.Dense(4, activation = "softmax")
    # keras.layers.Dropout(0.5),
    # keras.layers.Dense(10, activation='softmax')
])
m.summary()


# In[45]:



m.compile(optimizer= "adam", loss = 'categorical_crossentropy',
              metrics = [ TruePositives(name='tp'), 
                         FalsePositives(name='fp'), 
                         TrueNegatives(name='tn'), 
                         FalseNegatives(name='fn'), 
                         "accuracy", 
                         Precision(name='precision'), 
                         Recall(name='recall'), 
                         AUC(name='auc')])


# In[46]:


from keras.callbacks import TensorBoard, EarlyStopping
earlyStopping = EarlyStopping(monitor = 'loss', patience = 16, mode = 'min', restore_best_weights = True)


# In[47]:



history = m.fit(train_images, train_labels, epochs=20,validation_data=(test_images, test_labels), batch_size= 64,callbacks =[earlyStopping])     


# In[48]:


plt.plot(history.history['accuracy'], label='Training_Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation_Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

print(m.evaluate(test_images,  test_labels, verbose=2))



m.save('modelcnn_new.h5')


# In[50]:


import seaborn as sns

def plt_dynamic(x,ty,colors=['b']):
    ax.plot(x, ty, 'r', label="Train Loss")
    plt.legend()
    plt.grid()
    fig.canvas.draw()

def plt_dynamics(x,ty,colors=['b']):
    ax.plot(x, ty, 'b', label="Validation Loss")
    plt.legend()
    plt.grid()
    fig.canvas.draw()


# In[55]:



fig,ax = plt.subplots(1,1)
ax.set_xlabel('epochs') ; ax.set_ylabel('Categorical Crossentropy Loss')    
x = list(range(1,21))

ty = history.history['loss']
ty1=history.history['val_loss']
plt_dynamic(x,ty)
plt_dynamics(x,ty1)


# In[15]:


columns = ['Models','Testing Accuracies','Recall','Precision']
time_cnn=['Time Distributed Convolutional Neural Network',0.8139 ,0.7703 ,0.8482 ]
doc_net= ['Customized Convolutional Neural Network (DocNet)',0.97363, 0.9668,0.9798]
dd=pd.DataFrame([time_cnn,doc_net],columns=columns)
dd.head()

