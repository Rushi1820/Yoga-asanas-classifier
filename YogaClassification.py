#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import cv2
import re
import numpy as np


import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings('ignore')


# In[5]:


labels = []
images = []
asanas_name = []
images_path = []
images_pixels = []

i=0
dataset_path = "C:\\Users\\grvn1\\Downloads\\yoga"

for directory in os.listdir(dataset_path):
    asanas_name.append(directory)
    for img in os.listdir(os.path.join(dataset_path,directory)):  
        if len(re.findall('.png',img.lower())) != 0 or len(re.findall('.jpg',img.lower())) != 0 or len(re.findall('.jpeg',img.lower())) != 0:
            img_path = os.path.join(os.path.join(dataset_path,directory),img)
            images.append(img)
            images_path.append(img_path)
            img_pix = cv2.imread(img_path,1)
            images_pixels.append(cv2.resize(img_pix, (100,100)))
            labels.append(i)
            #print(f"{img} - {img_path}")
        
    i = i+1
print("Total labels: ", len(labels))
print("Total images: ", len(images))
print("Total images path: ", len(images_path))
print("Total asanas: ", len(asanas_name))
print("Total images_pixels: ", len(images_pixels))


# In[6]:


print(images_path[0:10])
print(images[0:10])


# In[7]:


fig = plt.gcf()
fig.set_size_inches(16, 16)

next_pix = images_path.copy()
random.shuffle(next_pix)

for i, img_path in enumerate(next_pix[0:12]):
    
    sp = plt.subplot(4, 4, i + 1)
    sp.axis('Off')

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()


# In[8]:


shuf = list(zip(images_pixels,labels))
random.shuffle(shuf)

train_data, labels_data = zip(*shuf)


# In[9]:


X_data = np.array(train_data) / 255
Y_data =  to_categorical(labels_data, num_classes = 107)


# In[10]:


Y_data[0]


# In[11]:


print("X data shape: ", X_data.shape)
print("Y data shape: ", Y_data.shape)


# In[12]:


X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size = 0.4, random_state=101)

print("X train data : ", len(X_train))
print("X label data : ", len(X_val))
print("Y test data : ", len(Y_train))
print("Y label data : ", len(Y_val))


# In[13]:


datagen = ImageDataGenerator(horizontal_flip=False,
                             vertical_flip=False,
                             rotation_range=0,
                             zoom_range=0.2,
                             width_shift_range=0,
                             height_shift_range=0,
                             shear_range=0,
                             fill_mode="nearest")


# In[14]:


pretrained_model = tf.keras.applications.DenseNet121(input_shape=(100,100,3),
                                                      include_top=False,
                                                      weights='imagenet',
                                                      pooling='avg')
pretrained_model.trainable = False


# In[15]:


inputs = pretrained_model.input
drop_layer = tf.keras.layers.Dropout(0.25)(pretrained_model.output)
x_layer = tf.keras.layers.Dense(512, activation='relu')(drop_layer)
x_layer1 = tf.keras.layers.Dense(128, activation='relu')(x_layer)
drop_layer1 = tf.keras.layers.Dropout(0.20)(x_layer1)
outputs = tf.keras.layers.Dense(107, activation='softmax')(drop_layer1)


model = tf.keras.Model(inputs=inputs, outputs=outputs)


# In[16]:


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['acc'])
history = model.fit(datagen.flow(X_train,Y_train,batch_size=32),validation_data=(X_val,Y_val),epochs=50)


# In[37]:


model.fit(X_train, Y_train)


# In[38]:


model.summary()


# In[17]:


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()


# In[18]:


plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()


plt.show()


# In[21]:


y_pred = model.predict(X_val)
y_pred = tf.argmax(y_pred, axis=1)
print(y_pred)


# In[22]:


y_test = tf.argmax(Y_val, axis=1)
y_test


# In[27]:


test = X_val[0].reshape(-1,100,100,3)
print(test.shape)
p = model.predict(test)

plt.imshow(X_val[0])
plt.title(asanas_name[np.argmax(p)])
np.argmax(p)
if np.argmax(Y_val[0]) == np.argmax(p):
    print("True prediction")
else:
    print("Wrong prediction")


# In[28]:


count = 0
for i in range(10):
    test_image = X_val[i].reshape(-1,100,100,3)
    res = Y_val[i]
    p = model.predict(test_image)
    
    
    if np.argmax(res) != np.argmax(p):
        plt.imshow(X_val[i])
        plt.title(asanas_name[np.argmax(p)])
        plt.show()
        print(f"True label: {asanas_name[np.argmax(res)]}")
        count = count + 1
        if count == 3:
            break


# In[ ]:




