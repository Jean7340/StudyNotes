#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["KERAS_BACKEND"] = "tensorflow"


import keras
from keras.layers import Dense, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Flatten, Reshape

import tensorflow as tf
import numpy as np

IMG_H, IMG_W = 64, 64
BATCH_SIZE = 256
D_LATENT = 128
N_FILTERS = (128, 128, 128, 128, 128)

#%% VAE

keras.saving.get_custom_objects().clear()
 
@keras.saving.register_keras_serializable()
class Sampler(keras.layers.Layer):
    def call(self, z_mean, z_logvar):
        epsilon = tf.random.normal( shape=z_mean.shape )
        return z_mean + tf.exp(0.5 * z_logvar) * epsilon

# Adding the @keras.saving.register_keras_serializable decorator to the class definition of a custom object registers the object globally in a master list, allowing Keras to recognize the object when loading the model.
# We can optionally specify a package or a name. If left blank, the package defaults to `Custom` and the name defaults to the class name.
@keras.saving.register_keras_serializable(name="Encoder")
class Encoder(keras.layers.Layer):
    def __init__(self, num_filters=N_FILTERS, d_latent=D_LATENT, d_proj=16, name="encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv_1 = Conv2D(num_filters[0], 3, strides=2, padding="same") # use stride instead of max pooling to downsample feature maps
        self.conv_2 = Conv2D(num_filters[1], 3, strides=2, padding="same") # strides are preferable for model that cares about location information which we do for image reconstruction       
        self.conv_3 = Conv2D(num_filters[2], 3, strides=2, padding="same")
        self.conv_4 = Conv2D(num_filters[3], 3, strides=2, padding="same")
        self.conv_5 = Conv2D(num_filters[4], 3, strides=2, padding="same")
        self.dense_mean = Dense(d_latent)
        self.dense_logvar = Dense(d_latent)
        
    def call(self, inputs):
        x = inputs
        for layer in [self.conv_1, self.conv_3, self.conv_3, self.conv_4, self.conv_5]:
            x = layer(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
        x = Flatten()(x)
        z_mean = self.dense_mean(x)
        z_logvar = self.dense_logvar(x)
        return z_mean, z_logvar
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "conv_1": keras.saving.serialize_keras_object(self.conv_1),
            "conv_2": keras.saving.serialize_keras_object(self.conv_2),
            "conv_3": keras.saving.serialize_keras_object(self.conv_3),
            "conv_4": keras.saving.serialize_keras_object(self.conv_4),
            "conv_5": keras.saving.serialize_keras_object(self.conv_5),
            "dense_mean": keras.saving.serialize_keras_object(self.dense_mean),
            "dense_logvar": keras.saving.serialize_keras_object(self.dense_logvar),
        }
        return {**base_config, **config}

        
@keras.saving.register_keras_serializable(name="Decoder")
class Decoder(keras.layers.Layer):
    def __init__(self, img_shape=(IMG_H,IMG_W), num_filters=N_FILTERS, d_latent=D_LATENT, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv_1 = Conv2DTranspose(num_filters[-1], 3, activation="relu", strides=2, padding="same") # revert Conv2D(64, 3) in the encoder; output shape (14,14,64)
        self.conv_2 = Conv2DTranspose(num_filters[-2], 3, activation="relu", strides=2, padding="same") # revert Conv2D(32, 3) in the encoder; output shape (28,28,32)
        self.conv_3 = Conv2DTranspose(num_filters[-3], 3, activation="relu", strides=2, padding="same")
        self.conv_4 = Conv2DTranspose(num_filters[-4], 3, activation="relu", strides=2, padding="same")
        self.conv_5 = Conv2DTranspose(num_filters[-5], 3, activation="relu", strides=2, padding="same")
        self.conv = Conv2DTranspose( 3, kernel_size=3, strides=1, activation="sigmoid", padding="same", name="conv")
        self.shape_before_flattening = np.array( [img_shape[0]/(2**len(num_filters)), img_shape[1]/(2**len(num_filters)), num_filters[-1]], dtype='i' )
        self.dense = Dense( np.prod(self.shape_before_flattening) )
        
    def call(self, latent_input):
        x = self.dense(latent_input)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Reshape( self.shape_before_flattening )(x)
        for layer in [self.conv_1, self.conv_3, self.conv_3, self.conv_4, self.conv_5]:
            x = layer(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x) 
        return self.conv(x)
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "conv_1": keras.saving.serialize_keras_object(self.conv_1),
            "conv_2": keras.saving.serialize_keras_object(self.conv_2),
            "conv_3": keras.saving.serialize_keras_object(self.conv_3),
            "conv_4": keras.saving.serialize_keras_object(self.conv_4),
            "conv_5": keras.saving.serialize_keras_object(self.conv_5),
            "conv": keras.saving.serialize_keras_object(self.conv),
            "dense": keras.saving.serialize_keras_object(self.dense),
        }
        return {**base_config, **config}
    
BETA = 1000

@keras.saving.register_keras_serializable(name="VAE")
class VAE(keras.Model):
    def __init__(self, img_shape=(IMG_H, IMG_W), num_filters=N_FILTERS, d_latent=D_LATENT, **kwargs):
        super().__init__(**kwargs)
        self.d_latent = d_latent
        self.encoder = Encoder(num_filters, d_latent)
        self.decoder = Decoder(img_shape=img_shape, num_filters=num_filters, d_latent=d_latent)
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
            total_loss = BETA * reconstruction_loss + tf.reduce_mean(kl_loss)
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
   
   
#%% VAE Model

# model_path = f'{os.getenv("HOME")}/Data/models/vae_celeba'
model_path = '/Users/liuzhiying/Desktop/CIS433_AI and Deep Learning/Codes/vae_celeba'
model = keras.saving.load_model( model_path + '.keras' ) # 25731.1738
d_latent = model.d_latent
