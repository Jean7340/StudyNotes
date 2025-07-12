#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
#os.environ["KERAS_BACKEND"] = "tensorflow"


import tensorflow as tf # tf.config.list_physical_devices('GPU')
import keras
print(f'keras version={keras.__version__}\nTensorflow version={tf.__version__}')

import numpy as np
from keras import Sequential
from keras.layers import Input, Dense, Reshape, Conv2D, SeparableConv2D, Dropout, MaxPooling2D, Flatten
from keras.layers import Normalization, BatchNormalization, LayerNormalization

#%% mnist 

import matplotlib.pyplot as plt
def plotEpoch( history, metric='acc'):    
    epochs = range(len(history['loss']))    
    if metric=='acc':
        plt.plot(epochs, history['accuracy'], 'k', label='Training acc')
        val = history['val_accuracy']
        plt.plot(epochs, val, 'b', label='Validation acc')
        plt.axvline(x=val.index(max(val)), color='r')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
    elif metric=='loss':
        plt.plot(epochs, history['loss'], 'k', label='Training loss')
        val = history['val_loss']
        plt.plot(epochs, val, 'b', label='Validation loss')        
        plt.axvline(x=val.index(min(val)), color='r')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()     

from keras.datasets import mnist
(images, labels), (test_images, test_labels) = mnist.load_data() # mnist.load_data() returns both training and test data
images = images.astype("float32") / 255 # convert integers from 0 to 255 to float from 0 to 1
test_images = test_images.astype("float32") / 255
train_size = int( images.shape[0]*0.8 ) # the 1st axis is the batch axis        

i = np.random.randint(0, high=test_images.shape[0])
plt.imshow( test_images[i], cmap=plt.cm.binary)
test_labels[i]


#%% Layer and Model
"""
A Layer object encapsulates both a state (the layer's "weights") and a transformation from inputs to outputs (a "call", the layer's forward pass)
A Model object is almost the same, except it has fit(), evaluate(), predict(), and saving methods which a Layer object doesn't have.
"""
def get_sequential_model(): 
    return Sequential([ Reshape((-1,)), Dense(512, activation="relu"), Dropout(0.5), Dense(10, activation="softmax") ])

model = get_sequential_model()
model.compile(optimizer="rmsprop", loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"]) # tfvariable_on_a_different_device error if RootMeanSquaredError() or mse is included
history = model.fit( images, labels, epochs=10, batch_size=128, validation_split=0.2 ) # 20% for validation
plotEpoch( history.history, metric='acc' )

model.summary() # since we did not specify the input shape, we have to run summary() after fitting the model with data
keras.utils.plot_model(model, show_shapes=True) # apt install graphviz


model.evaluate( test_images, test_labels ) # return a list of 2 scalars: loss, accuracy
predictions = model.predict( test_images, batch_size=128 )

i = np.random.randint(0, high=test_images.shape[0])
plt.imshow( test_images[i], cmap=plt.cm.binary)
predictions[i] # 10 probabilities, corresponding to 0, 1, ..., 9


#%%  Functional API

# (batch size, 28, 28) ->  (batch size, 28, 28, 1)
train_ds = tf.data.Dataset.from_tensor_slices( (np.expand_dims(images[:train_size], axis=3), labels[:train_size]) ).batch(128)
val_ds   = tf.data.Dataset.from_tensor_slices( (np.expand_dims(images[train_size:], axis=-1), labels[train_size:]) ).batch(128)
test_ds  = tf.data.Dataset.from_tensor_slices( (np.expand_dims(test_images, -1), test_labels) ).batch(128)

def get_functional_model():
    inputs = Input( shape=(28,28,1) ) # the shape argument is the shape of one instance without the batch dimension
    x = Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(64, 3, activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = SeparableConv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = Flatten()(x)
    outputs = Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = get_functional_model()
model.compile( optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"] )
callbacks_list = [ keras.callbacks.EarlyStopping( monitor="val_accuracy", patience=2 ), keras.callbacks.ModelCheckpoint( filepath="/tmp/mnist.keras", monitor="val_loss", save_best_only=True )   ]
model.fit( train_ds, epochs=5, callbacks=callbacks_list, validation_data=val_ds ) # 98.6% at epoch 5; stop around epoch 11
model.evaluate( test_ds ) # return a list of 3 scalars: loss, accuracy, rmse 
    
#%% Under the hood: Forward pass + Backpropagation

model = get_functional_model()
loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.RMSprop()
metrics = [ keras.metrics.SparseCategoricalAccuracy() ]
loss_tracker = keras.metrics.Mean()
"""
Forward pass:  use model() with "training=True" since some layers (e.g., Dropout) behave differently during training and inference
Backward pass: retrieve gradient of trainable weights only. Non-trainable weights (e.g., BatchNormalization) are meant to be updated during the forward pass by the layers owning them
"""
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients( zip(gradients, model.trainable_weights) ) # apply_gradients() expects a list of (gradient, variable) pairs. 
    logs = {}
    for metric in metrics:
        metric.update_state(targets, predictions)
        logs[metric.name] = metric.result()
    loss_tracker.update_state(loss)
    logs["loss"] = loss_tracker.result()
    return logs
""" 
For inference, there is no backward pass and we need to set "training=False" in the forward pass.
Adding the tf.function decorator to compile the code into a computation graph for optimization.
"""
@tf.function 
def test_step(inputs, targets):
    predictions = model(inputs, training=False) 
    loss = loss_fn(targets, predictions)
    logs = {}
    for metric in metrics:
        metric.update_state(targets, predictions)
        logs["val_" + metric.name] = metric.result()
    loss_tracker.update_state(loss)
    logs["val_loss"] = loss_tracker.result()
    return logs

def reset_metrics(): # reset the state of all metrics at the start of each epoch
    for metric in metrics:
        metric.reset_state()
    loss_tracker.reset_state()

# This is what happens when you run model.fit()
for epoch in range(2):
    reset_metrics()
    batch_seen = 0
    for inputs_batch, targets_batch in train_ds:
        batch_seen += 1
        print(f"Processing batch {batch_seen} of epoch {epoch}")
        logs = train_step(inputs_batch, targets_batch)
    print(f"Results at the end of epoch {epoch}")
    for key, value in logs.items():
        print(f"...{key}: {value:.4f}")
    # Under the hood of evaluate()
    reset_metrics()
    for inputs_batch, targets_batch in val_ds:
        logs = test_step(inputs_batch, targets_batch)
    print("Evaluation results:")
    for key, value in logs.items():
        print(f"...{key}: {value:.4f}")


#%% Nature of Overfitting

images_with_noise = np.concatenate( [images, np.random.random(images.shape)], axis=1)
images_with_zeros = np.concatenate( [images, np.zeros(images.shape)], axis=2) # axis=1 or 2 doesn't matter after flattening
images_with_noise.shape, images_with_zeros.shape

i = np.random.randint(0, high=images_with_noise.shape[0])
labels[i]

plt.imshow( images_with_noise[i], cmap=plt.cm.binary)
plt.imshow( images_with_zeros[i], cmap=plt.cm.binary)

n_epochs = 20

model = get_sequential_model()
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history_noise = model.fit( images_with_noise, labels, epochs=n_epochs, batch_size=128, validation_split=0.2 ) 

model = get_sequential_model()
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history_zeros = model.fit( images_with_zeros, labels, epochs=n_epochs, batch_size=128, validation_split=0.2 )

plt.plot(range(n_epochs), history_noise.history["val_accuracy"], "b-",  label="Validation accuracy with noise channels")
plt.plot(range(n_epochs), history_zeros.history["val_accuracy"], "b--", label="Validation accuracy with zeros channels")
plt.title("Effect of noise channels on validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

random_train_labels = labels[:train_size].copy() # unlike a list, numpy array slices are views on the original array, NOT a copy!
np.random.shuffle( random_train_labels )
model = get_sequential_model()
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history_random = model.fit( images[:train_size], random_train_labels, epochs=100, batch_size=128, validation_data=(images[train_size:], labels[train_size:]) )
plotEpoch( history_random.history )


#%% Special layers: Normalization, Batch Normalization, Layer Normalization
"""
Batch Normalization layer has different behavior in training vs. inferencing:
1) During training, the layer normalizes its output using the mean and standard deviation of the current batch of inputs.
2) During inference, the layer normalizes its output using a moving average of the mean and standard deviation of the batches it has seen during training.
BatchNormalization() returns gamma*(batch-mean(batch))/sqrt(var(batch)+epsilon)+beta where gamma and beta initialize to 1 and 0, respectively.
"""
x = np.array([ [1,3,2], 
               [3,3,3], 
               [5,3,4] ], dtype='float32')

layer_n = Normalization(axis=None)
adapt_data = np.array([1, 5], dtype='float32')
np.mean(adapt_data), np.std(adapt_data)
layer_n.adapt(adapt_data)
layer_n(x)
( x - np.mean(adapt_data) ) / np.std(adapt_data)

epsilon = 0.002
layer_bn = BatchNormalization(epsilon=epsilon)
layer_bn( x, training=True )      # the default is training=False
( x - np.mean(x, 0) )/np.sqrt( np.var(x, 0) + epsilon )

""" Layer normalization computes, for each training instance, the normalization statistics over all neurons of the same layer """
layer_ln = LayerNormalization(axis=1)
x = tf.constant(np.arange(10).reshape(5, 2) * 10)
x, layer_ln(x)