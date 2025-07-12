#!/usr/bin/env python3

import tensorflow as tf
tf.__version__
gpus = tf.config.list_physical_devices('GPU') # list available GPU
for i, gpu in enumerate(gpus):
    print(f'GPU {i}:')
    print( tf.config.experimental.get_device_details(gpu) )
    print( tf.config.experimental.get_memory_info(f'GPU:{i}') )

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
print(f'keras version={keras.__version__}\nTensorflow version={tf.__version__}')

import numpy as np

#%%

tf.constant( [[1, 2, 3], [4, 5, 6]], dtype=tf.int32 ) # Create a constant tensor from a tensor-like object
tf.zeros( [2, 3] )

m1 = tf.constant( 1, shape=(2, 3) ) # If dtype is unspecified, the type is inferred from the values
    
m2 = tf.ones( [1, 3], tf.int32 )
tf.matmul(m1, m2) # InvalidArgumentError: Matrix size-incompatible

m2 = tf.ones( [3, 1], tf.int32 )
tf.matmul(m1, m2)

# TensorFlow doesn't treat a vector as a matrix
m2 = tf.constant( [1,1], tf.int32 )
tf.matmul(m2, m1) # InvalidArgumentError: In[0] and In[1] has different ndims

tf.fill( [3, 3], 9 ) # fill with a scalar value, same as tf.constant(9, shape=(3,3))
tf.eye(3) # Construct an identity matrix, or a batch of matrices
tf.eye(2, num_columns=3, dtype=tf.uint8)
tf.range(start=3, limit=18, delta=3) # Like the Python builtin range(), start defaults to 0

tf.where([True, False, False, True], x=[1,2,3,4], y=[100]) # return elements from x or y depending on the condition

tf.random.normal(  shape=[2,2], mean=0.0, stddev=1.0)
tf.random.normal( [2,2] )

t = tf.random.uniform( shape=[3,4], minval=0, maxval=9, dtype=tf.dtypes.int32 )
t[0,1] = 9 # TypeError: unlike ndarray, tensors are immutable

t.shape
t.numpy()
tf.size(t) # returns a 0-D tensor representing the number of elements in the input tensor
tf.rank(t) # returns a 0-D int32 tensor representing the rank of the input tensor
tf.reshape(t, [4, 3])  # returns a new tensor holding the same values in the same order but with a new shape
tf.reshape(t, 12)      # tf.reshape() does not change the order of or the total number of elements in the tensor
tf.reshape(t, [2, -1]) # If one element of the shape argument is -1, the size of that dimension is computed to maintain size
tf.reshape(t, -1)      # In particular, a shape argument of [-1] flattens the tensor.

tf.unique( tf.reshape(t, -1) ) 
tf.zeros_like( t ) # returns a tensor of the same type and shape with all elements set to zero
tf.ones_like( t, dtype=tf.float32 )
tf.cast( t, tf.float16 ) # Casts a tensor to a new type

tf.shape_n([ tf.cast(t, tf.float32), tf.ones([1, 2]) ]) # returns the shapes of multiple tensors in a list

t_batched = tf.expand_dims(t, axis=0) # insert a dimension of length 1 at the given axis 
t_batched.shape                       # TensorShape([1, 3, 4])  
t_batched_channeled = tf.expand_dims(t_batched, -1)  # negative index means counting backward from the end
t_batched_channeled.shape

tf.squeeze( t_batched_channeled )
tf.squeeze( t_batched, axis=0)

#%% Function |Purpose                         |Result Shape      |Input Shape               |Axis/Dims       
### ---------|--------------------------------|------------------|--------------------------|----------------
### tf.repeat|Repeats each element of a tensor|Varies            |Same as input tensor      |Scalar or Tensor
### tf.stack |Stacks tensors along a new axis |Higher rank tensor|Same for all input tensors|New Axis        
### tf.concat|Concatenates tensors            |Varies            |Same for all input tensors|Existing Axis   
### tf.tile  |Replicates tensor(s)            |Larger size       |Same as input tensor(s)   |Specified Dims  
### ---------|--------------------------------|------------------|--------------------------|----------------

tf.repeat( ['a', 'b', 'c'], repeats=[3, 0, 2], axis=0 ) # repeats each element of a tensor certain number of times along a given axis
tf.repeat( [[1,2,3], [4,5,6]], repeats=[2, 3], axis=0 ) 

# tf.stack() is used to stack multiple tensors along a new axis. All tensors must have the same shape for all dimensions except the one along which they are being stacked.
x, y, z = tf.constant([1, 4]), tf.constant([2, 5]), tf.constant([3, 6])
t = tf.stack([x, y, z], axis=0) # stack a list of tensors into a tensor whose rank is one higher than each input tensor
# Given a list of N tensors of shape (A, B, C), if `axis = 0`, the output tensor will have the shape (N, A, B, C)
tf.unstack(t) # Unpack by chipping the input tensor along the given axis which defaults to 0
tf.unstack(t, axis=1) 

# tf.concat() is used to concatenate tensors along an existing axis. All tensors must have the same shape for all dimensions except the one along which they are being concatenated.
p, q = [[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]
tf.concat( [p, q], axis=0 ) # Concatenate tensors along an existing axis to output a tensor whose rank is the same as each input tensor
tf.concat( [p, q], axis=1 ) # unlike stack(), axis has no default and must be provided

# tf.tile(input, multiples) replicates the entire tensor along given axes. So len(multiples) must be the same as the rank of input
t = tf.tile( p, multiples=[2, 3] ) # The values of input are replicated multiples[i] times along the i-th dimension

# tf.split(value, num_or_size_splits, axis=0) splits a tensor into a list of sub tensors
t = tf.stack( [tf.range(1,11), tf.range(11,21)], axis=0 ) # shape=(2, 10) 
tf.split(t, num_or_size_splits=2, axis=1) # num_or_size_splits is either an integer or a 1D tensor
tf.split(t, [1, 4, 5], axis=1) # 1 + 4 + 5 == 10

#%% TensorFlow Variable

w = tf.Variable( [[1,2],[3,4]], name='weight_layer1')
w.shape, w.name, w.value()
tf.add(w, tf.ones_like(w) )
tf.pow(w, w)

t1 = tf.ones( [2, 3, 4, 5] )
t2 = tf.ones( [2, 3, 5, 4] )
tf.matmul(t1, t2).shape

t3 = tf.ones( [5, 6] )
tf.tensordot(t1, t3, 1).shape # tensordot(A,B,N), aka tensor contraction, calculates the sumproduct of elements over the last N axes of A and the first N axes of B in order.

#%% Loss

y_true = [1, 2] 
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]] 
scce = keras.losses.SparseCategoricalCrossentropy()
scce( np.array(y_true), np.array(y_pred) )


y_true = [ [1, 2], 
           [2, 1] ]
y_pred = [ [[0.05, 0.95, 0], [0.1, 0.8, 0.1]], 
           [[0.1, 0.8, 0.1], [0.05, 0.95, 0]] ]
scce( np.array(y_true), np.array(y_pred) )


y_true = [ [[ 0,  2],
           [-1, -1]],
           [[ 0,  2],
           [-1, -1]] ]
y_pred = [ [[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.2, 0.5, 0.3], [0.0, 1.0, 0.0]]],
           [[[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]],
           [[0.2, 0.5, 0.3], [0.0, 1.0, 0.0]]] ]
scce = keras.losses.SparseCategoricalCrossentropy(ignore_class=-1)
scce( np.array(y_true), np.array(y_pred) )


#%% Metric

class RootMeanSquaredError(keras.metrics.Metric): # keras.metrics.Metric inherits from keras.layers.Layer

    def __init__(self, name="rmse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mse_sum = self.add_weight(name="mse_sum", initializer="zeros") # adds a new variable to the layer
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros", dtype="int32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.one_hot( y_true, depth=tf.shape(y_pred)[1] ) # true label as a one-hot vector; y_pred is a batch of prob vectors
        mse = tf.reduce_sum( tf.square(y_true - y_pred) )
        self.mse_sum.assign_add( mse )
        self.total_samples.assign_add( tf.shape(y_pred)[0] ) # add number of samples

    def result(self):
        return tf.sqrt( self.mse_sum / tf.cast(self.total_samples, tf.float32) )

    def reset_state(self):
        self.mse_sum.assign(0.)
        self.total_samples.assign(0)

mymetric = RootMeanSquaredError()
metric = keras.metrics.SparseCategoricalAccuracy()

targets = [0, 1, 2, 0]
predictions = [ [1,0,0], [0,1,0], [0,0,1], [0,0,1] ] # correct 3 out of 4
metric.update_state( targets, predictions )
mymetric.update_state( targets, predictions )

metric.result() # 0.75
mymetric.result() # sqrt(  (1+1)/4 ) = 0.7071

predictions = [ [0,1,0], [1,0,0], [0,0,1], [0,0,1] ] # correct 1 out of 4
metric.update_state( targets, predictions )
metric.result() # 0.5
metric.reset_state()
predictions = [ [0,1,0], [1,0,0], [0,0,1], [0,0,1] ] # correct 1 out of 4
metric.update_state( targets, predictions )
metric.result() # 0.25

mean_tracker = keras.metrics.Mean()
for v in range(1,10,2):
    mean_tracker.update_state(v)
mean_tracker.result()
        


