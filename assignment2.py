#!/usr/bin/env python
# coding: utf-8

# In[5]:


import talos
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import backend
import pickle
p = {
    'units': [120, 240],
    'hidden_activations': ['relu', 'sigmoid'],
    'activation': ['softmax', 'sigmoid'],
    'loss': ['mse', 'categorical_crossentropy'],
    'optimizer': ['adam', 'adagrad'],
    'batch_size': [1000, 2000]
}


# In[8]:


with open('train', 'rb') as file:
    train_dict = pickle.load(file, encoding='bytes')

with open('test', 'rb') as file:
    test_dict = pickle.load(file, encoding='bytes')
    
x_train = train_dict[b'data']
y_train = train_dict[b'coarse_labels']
x_test = test_dict[b'data']
y_test = test_dict[b'coarse_labels']
y_train = tf.keras.utils.to_categorical(y_train, 100)
y_test = tf.keras.utils.to_categorical(y_test, 100)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train=x_train/255.0
x_test=x_test/255.0
#x_train = x_train.reshape(-1, 3072)
#x_test = x_test.reshape(-1, 3072)


# In[11]:


def my_model(x_train, y_train, x_val, y_val, params):

    model = Sequential()
    

    model.add(Dense(units=params['units'],input_shape=(3072,))) #32X32px 3RGB so 32x32x3 = 3072
    model.add(Dense(units=params['units'], activation=params['hidden_activations']))
    model.add(Dense(units=params['units'], activation=params['hidden_activations']))
    model.add(Dense(units=params['units'], activation=params['hidden_activations']))
    model.add(Dropout(.2))
    model.add(Dense(units=params['units'], activation=params['hidden_activations']))
    model.add(Dense(units=params['units'], activation=params['hidden_activations']))
    model.add(Dense(units=params['units'], activation=params['hidden_activations']))
    model.add(Dense(units=100, activation=params['hidden_activations']))
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])

    out = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    batch_size=params['batch_size'],
                    epochs=200,
                    verbose=1)
    return out, model

talos.Scan(x_train, y_train, p, my_model, x_val=x_test, y_val=y_test, experiment_name="talos_output")


# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
r = talos.Reporting('talos_output/accuracy.csv')
r.table("accuracy")


# In[ ]:





# In[ ]:




