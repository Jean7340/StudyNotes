#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf # tf.config.list_physical_devices('GPU')
import keras
from keras.layers import Input, Dense, Conv2D, Reshape, Flatten, Conv2DTranspose

IMG_H, IMG_W = 28, 28
BATCH_SIZE = 256
D_LATENT = 2

#%% Variational AutoEncoder

encoder_input = Input( shape=(IMG_H, IMG_W, 1) )
x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_input) # use stride instead of max pooling to downsample feature maps
x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x) # strides are preferable for model that cares about location information which we do for image reconstruction
shape_before_flattening = x.shape[1:]
x = Flatten()(x) # output shape: 3136 = 7*7*64 where 7=(28/2)/2 after downsampling twice by strides=2
x = Dense(16, activation="relu")(x)
z_mean = Dense(D_LATENT, name="z_mean")(x)
z_logvar = Dense(D_LATENT, name="z_logvar")(x)
encoder = keras.Model(encoder_input, [z_mean, z_logvar], name="encoder")

latent_input = Input( shape=(D_LATENT,) )
x = Dense( np.prod(shape_before_flattening), activation="relu")(latent_input) # output a vector of size we had at the Flatten layer in the encoder
x = Reshape( shape_before_flattening )(x) # revert the Flatten layer in the encoder
x = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x) # revert Conv2D(64, 3) in the encoder; output shape (14,14,64)
x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x) # revert Conv2D(32, 3) in the encoder; output shape (28,28,32)
decoder_output = Conv2D(1, 3, activation="sigmoid", padding="same")(x) # output shape (28,28,1)
decoder = keras.Model(latent_input, decoder_output, name="decoder")

class Sampler(keras.layers.Layer):
    def call(self, z_mean, z_logvar):
        epsilon = tf.random.normal( shape=z_mean.shape )
        return z_mean + tf.exp(0.5 * z_logvar) * epsilon

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = Sampler()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
     # https://www.freecodecamp.org/news/python-property-decorator/
    @property # this decorator implements a getter for the protected attributes
    def metrics(self): # list the metrics here to enable the model to reset them after each epoch
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data): # this is how you override the train_step() method of the keras.Model class
        with tf.GradientTape() as tape:
            z_mean, z_logvar, z, reconstruction = self(data, training=True)
            reconstruction_loss = tf.reduce_mean( # average over the batch dimension
               tf.reduce_sum( keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2) ) # sum over the spatial dimensions.
            )
            kl_loss = -0.5 * (1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)) # regularization: K-L divergence from N(0,1)
            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
        grads = tape.gradient( total_loss, self.trainable_weights ) # retrieve the gradients; Use trainable_weights!
        self.optimizer.apply_gradients( zip(grads, self.trainable_weights) ) # apply_gradients() expects a list of (gradient, variable) pairs.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return { "total_loss": self.total_loss_tracker.result(),
                 "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                 "kl_loss": self.kl_loss_tracker.result()        }
    
    def call(self, inputs):
        z_mean, z_logvar = self.encoder(inputs)
        z = self.sampler(z_mean, z_logvar)
        reconstruction = self.decoder(z)
        return z_mean, z_logvar, z, reconstruction
       

#%%

# os.chdir(f'{os.getenv("HOME")}/Data')

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_all = np.concatenate( [x_train, x_test], axis=0 ) # train on all MNIST digits. Note train_step() doesn't expect any label
x_all = np.expand_dims(x_all, -1).astype("float32") / 255 # add channel as the last axis

vae = VAE(encoder, decoder)
vae.compile( optimizer=keras.optimizers.Adam(), run_eagerly=True ) # no specification of loss since it's taken care in train_step() of our customerized model VAE
vae.fit( x_all, epochs=10, batch_size=BATCH_SIZE ) # Epoch 30: kl_loss=3, reconstruction_loss=33, total_loss=36
    
#%% Generate new images

n = 30 # display a grid of 2n by n digits by sampling points linearly on a 2D grid
grid_x = np.linspace(-2, 2, 2*n) # return evenly spaced numbers over the interval
grid_y = np.linspace(-2, 2, n)
digit_size = 28
figure = np.zeros( (digit_size * len(grid_y), digit_size * len(grid_x)) )
for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array( [[xi, yi]] ) # batch of 1
        x_decoded = vae.decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[ i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size ] = digit
        
figure = 1 - figure # black digits on white background
plt.figure( figsize=(30, 10) )
plt.axis("off")
plt.imshow(figure, cmap="Greys_r")

#%% Explore the latent space

n = 5000
z_mean, z_logvar = vae.encoder.predict( x_all[:n] ) # both z_mean and z_logvar are of shape (5000, 2)
z = Sampler()(z_mean, z_logvar ) # shape: (5000, 2)
y_all = np.concatenate( [y_train, y_test], axis=0 )

plt.figure( figsize=(10, 10) )
plt1 = plt.scatter(z[:,0], z[:,1], cmap="rainbow", c=y_all[:n], alpha=0.8, s=3) # plt.scatter(z_mean[:,0], z_mean[:,1], cmap="rainbow", c=y_all[:n], alpha=0.8, s=3)
plt.colorbar(plt1)
plt.show()