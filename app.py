#!/usr/bin/env python
# coding: utf-8

# In[1]:


#libs
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

#data directories

DATADIR  = "/home/spidy/Documents/RJIT/PicData"
CATEGORIES = ["sfw", "nsfw"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category) #path to sfw and nfsw dir
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap ="gray")
        plt.show()
        break
    break


# In[2]:


print (img_array).shape


# In[3]:


IMG_SIZE = 80
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap = 'gray')
plt.show()


# In[4]:


training_data = []
def create_training_data():
    for category in  CATEGORIES:
        path = os.path.join(DATADIR, category) #path to sfw and nsfw
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
            
        
create_training_data()


# In[5]:


print(len(training_data))


# In[6]:


import random 
random.shuffle(training_data)


# In[7]:


for sample in training_data[:10]:
    print(sample[1])


# In[8]:


X = []
y = []



# In[9]:


for features, label in training_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# In[10]:


import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[11]:


pickle_in = open("X.pickle", "rb")
X =  pickle.load(pickle_in)
                 


# In[13]:


X[1]


# In[14]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255.0
y = np.array(y)

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", 
             optimizer="adam",
             metrics=['accuracy'])

model.fit(X, y, batch_size=8, epochs=8, validation_split=0.1)





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




