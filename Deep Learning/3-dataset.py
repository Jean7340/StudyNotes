#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf # tf.config.list_physical_devices('GPU')
import keras

#%% Training/Validation data is passed to fit() either as 2 Numpy arrays (inputs and targets) or as a
### TensorFlow Dataset object --- a generator that returns batches of features and labels
""" 
from_tensor_slices() creates a Dataset whose elements are slices of the input --- a tensor with the same size in their first dimension which becomes the dataset dimension. 
The given tensors are sliced along their first dimension, preserving the structure of the input tensors while removing that first dimension of each tensor.
"""
# Slicing a 1D tensor produces scalar tensor elements.
ds = tf.data.Dataset.from_tensor_slices( [1, 2, 3] )
I = iter(ds)
I is ds # False
next(I)
# Slicing a 2D tensor produces 1D tensor elements.
ds = tf.data.Dataset.from_tensor_slices( [[1,2], [3,4], [5,6]] )
I = iter(ds)
next(I)

# Slicing a tuple or a dict differs from slicing a tensor
ds = tf.data.Dataset.from_tensor_slices( ([1,2], [3,4], [5,6]) )
I = iter(ds)
next(I)
ds = tf.data.Dataset.from_tensor_slices( {"a": [1, 2], "b": [3, 4]} )
I = iter(ds)
next(I) # dict structure is preserved

features = tf.constant( [[1, 3], [2, 1], [3, 3]] ) # 3x2 tensor
labels = tf.constant(['A', 'B', 'C']) # 3x1 tensor
ds = tf.data.Dataset.from_tensor_slices( (features, labels) )
I = iter(ds)
next(I)
I = iter( ds.take(2) )
next(I)

labels2 = tf.constant(['a', 'b', 'c'])
ds = tf.data.Dataset.from_tensor_slices( {"features":features, "labels":labels, "labels2":labels2} )
I = iter(ds)
next(I)

# from_tensors() create a Dataset with a single element, rarely used in our course.
ds = tf.data.Dataset.from_tensors( ([1, 2, 3], 'A') ) 
list(ds.as_numpy_iterator())

#%% batching and mapping

arr = np.arange(11)[:, np.newaxis] # shape: (12,1)
features = np.concatenate( (arr, arr+10), axis=1 ) # shape: (12,2)
labels = np.squeeze( arr % 2) # np.squeeze removes axes of length one
ds = tf.data.Dataset.from_tensor_slices( (features, labels) )
I = iter(ds)
next(I)

ds.take(1).get_single_element()
ds.batch(3).take(1).get_single_element() # a single batch as a tuple of (x,y)

for x,y in ds.batch(3): # loop over all batches
    print("data batch shape:", x.shape)   # (3, 2)
    print("labels batch shape:", y.shape) # (3,)
    break

# Don't batch more than once!
for x,y in ds.batch(3).batch(2): # 
    print("data batch shape:", x.shape)   # (2, 3, 2)
    print("labels batch shape:", y.shape) # (2, 3)
    break

for x,y in ds.batch(3).take(2): # loop over 2 batches each with 3 elements
    print("data batch shape:", x.shape)   # (3, 2)
    print("labels batch shape:", y.shape) # (3,)


I = iter( ds.repeat(3).batch(2) )
next(I)

# tf.data.Dataset.map(f) produces a new dataset by applying f to the input dataset element by element
# f takes a tf.Tensor object representing an element in the input, and returns a tf.Tensor object representing an element in the new dataset

I = iter( ds.map( lambda x,y: (x-10, y) ) )
next(I)

def preprocess(x,y):
    z = x + np.array([4,5])
    return z,y

I = iter( ds.map( preprocess, num_parallel_calls=4 ) )
next(I)

#%% Tabular Data 

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Input, CategoryEncoding, StringLookup, Normalization, Embedding, Concatenate, Dense

# os.chdir(f'{os.getenv("HOME")}/Data/csv')
df = pd.read_csv("titanic.csv" ) # https://www.kaggle.com/datasets/yasserh/titanic-dataset
df.head(1) 
df.dtypes # read_csv() automatically detects data types, unless you have some inconsistent rows
df['sibsp'].unique() # [0,1,2,3,4,5,8], num of siblings/spouses aboard, not suitable for CategoryEncoding() which requires 0 <= value < num_tokens
df['parch'].unique() # [0,1,2,3,4,5,6], num of parents/children aboard

df['target'] = np.where(df['survived']==0, 0, 1) # we could have used df['survived'] directly. This is only for illustration.
df = df.drop(columns=['survived'])

train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False) # Set "shuffle=False" for ease of illustration. Default value is 'True'
train_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=False)
len(train_df), len(val_df), len(test_df)

x = df['sex'][:3].to_numpy()
StringLookup(vocabulary=df['sex'].unique(), output_mode='one_hot', num_oov_indices=0)(x) # set the number of out-of-vocabulary tokens to 0, which produces an error for OOV inputs
StringLookup(vocabulary=df['sex'].unique(), output_mode='one_hot')(x)

def df_to_dataset(df, shuffle=True, batch_size=32):
    dfc = df.copy()
    labels = dfc.pop('target')
    dfc = {key: value.to_numpy() for key, value in dfc.items()}
    ds = tf.data.Dataset.from_tensor_slices( (dfc, labels) )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dfc))
    ds = ds.batch(batch_size)    
    return ds

batch_size = 64
train_ds = df_to_dataset( train_df, shuffle=False, batch_size=batch_size )
val_ds = df_to_dataset(  val_df, shuffle=False, batch_size=batch_size )
test_ds = df_to_dataset( test_df, shuffle=False, batch_size=batch_size )

next(iter(train_ds))[0] # an example batch

for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of sibsp:', feature_batch['sibsp'])
    print('A batch of targets:', label_batch)

raw = {'sex': Input(shape=(), dtype='string', name="sex"),
       'age': Input(shape=(), dtype='int64', name="age"),
       'sibsp': Input(shape=(1,), dtype='int64', name="sibsp"),
       'parch': Input(shape=(), dtype='int64', name="parch"),
       'fare': Input(shape=(1,), dtype='float32', name="fare"),
       'class': Input(shape=(), dtype='string', name="class"), 
       'deck': Input(shape=(), dtype='string', name="deck"),
       'embarked': Input(shape=(), dtype='string', name="embarked"),
       'alone': Input(shape=(), dtype='string', name="alone")}

features = {'sex': StringLookup(vocabulary=df['sex'].unique(), output_mode='one_hot', num_oov_indices=0)(raw['sex']), # number of out-of-vocabulary tokens to use, default is 1
            'age': raw['age'],
            'sibsp': Normalization( axis=None, mean=df['sibsp'].mean(), variance=df['sibsp'].var() )(raw['sibsp']),
            'parch': CategoryEncoding( num_tokens=len(df['parch'].unique()), output_mode='one_hot')(raw['parch']), 
            'fare': Normalization( axis=None, mean=df['fare'].mean(), variance=df['fare'].var() )(raw['fare']), 
            'class': StringLookup(vocabulary=df['class'].unique(), output_mode='one_hot', num_oov_indices=0)(raw['class']), 
            'deck': StringLookup(vocabulary=df['deck'].unique(), output_mode='one_hot', num_oov_indices=0)(raw['deck']),
            'embarked': StringLookup(vocabulary=df['embarked'].unique(), output_mode='one_hot', num_oov_indices=0)(raw['embarked']),
            'alone': StringLookup(vocabulary=df['alone'].unique(), output_mode='one_hot', num_oov_indices=0)(raw['alone'])}

preprocessor = keras.Model(inputs=raw, outputs=features)
"""
An Embedding layer turns positive integers (indexes) into dense vectors of fixed size.
"""
embedding = Embedding(len(df['age'].unique()), 2)
x = preprocessor(raw)
x = Concatenate()( [x['sex'], embedding(x['age']), x['sibsp'], x['parch'], x['fare'], x['class'], x['deck'], x['embarked'], x['alone'] ] )
outputs = Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs=raw, outputs=outputs)

"""
Default activation of Dense(1) is 'None' in which case we should use the loss function BinaryCrossentropy(from_logits=True).
It expects y_pred of shape [batch_size] and converts the logistic units to probabilities.
Alternatively, we can use Dense(2, activation='softmax') which outputs a vector of probabilities that sum to 1.
In that case, use the loss function SparseCategoricalCrossentropy() that expects y_pred of shape [batch_size, num_classes].
"""
model.summary()
keras.utils.plot_model(model)
model.compile( loss='BinaryCrossentropy', metrics=["accuracy"])
model.fit( train_ds, epochs=20, validation_data=val_ds )
model.evaluate( test_ds )

# Predict the survival of Rose who is rescued and later reunited with Jack in the afterlife after 84 years 
Rose = {'sex':['female'], 'age':[17], 'sibsp':[0], 'parch':[3], 'fare': [70.25], 'class': ['First'], 'deck':['A'], 'embarked':['Southampton'],'alone':['n']}
Rose_Dataseted = tf.data.Dataset.from_tensor_slices(Rose).batch(1)
model.predict(Rose_Dataseted)

#%% Image Tensor: (height, width, channel)

from keras.utils import image_dataset_from_directory

# os.chdir(f'{os.getenv("HOME")}/Data')
base_dir = 'dogcat/small'
# By default, labels='inferred', meaning labels are inferred from directory structure
# By default, interpolation='bilinear', which is the interpolation method used for resizing images
ds = image_dataset_from_directory( base_dir+"/train",  image_size=(180, 180), batch_size=32 )

import matplotlib.pyplot as plt
for imgs, labels in ds:
    print(imgs.shape)
    plt.imshow( tf.cast(imgs[0], tf.uint8) )     # imshow() expects 0-1 float or 0-255 int
    plt.imshow( imgs[0].numpy().astype('uint8')) # alternatively, convert to numpy type uint8
    plt.title( f'label={labels[0].numpy()}' )
    break
       
#%% Text Data

import string
import re

class Vectorizer:
    def standardize(self, text):
        text = text.lower()
        return "".join(char for char in text if char not in string.punctuation)

    def tokenize(self, text):
        text = self.standardize(text)
        return text.split()

    def adapt(self, dataset):
        self.vocabulary = {"": 0, "[UNK]": 1} # index 0 is for mask token; 1 is for OOV token
        for text in dataset:
            tokens = self.tokenize(text)
            for token in tokens:
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)
        self.inverse_vocabulary = dict( (v, k) for k, v in self.vocabulary.items() ) # dict(generator expression) with omitted parentheses, same as {v:k for k,v in self.vocabulary.items()}

    def encode(self, text):        
        tokens = self.tokenize(text)
        return [self.vocabulary.get(token, 1) for token in tokens]

    def decode(self, int_sequence):
        return " ".join( self.inverse_vocabulary.get(i, "[UNK]") for i in int_sequence )

simple_vectorizer = Vectorizer()
dataset = ["I write, erase, rewrite", "Erase again, and then", "A poppy blooms."]
simple_vectorizer.adapt(dataset)

test_sentence = "I write, rewrite, and still rewrite again"

encoded_sentence = simple_vectorizer.encode( test_sentence )
decoded_sentence = simple_vectorizer.decode(encoded_sentence)

from keras.layers import TextVectorization, Embedding

def custom_standardization_fn(string_tensor):
    lowercase_string = tf.strings.lower(string_tensor)
    return tf.strings.regex_replace( lowercase_string, f"[{re.escape(string.punctuation)}]", "")

def custom_split_fn(string_tensor):
    return tf.strings.split(string_tensor)

vectorizer = TextVectorization( output_mode="int",  standardize=custom_standardization_fn, split=custom_split_fn )
vectorizer.adapt(dataset)

vocabulary = vectorizer.get_vocabulary()
vectorizer( test_sentence )
int2wrd = dict(enumerate(vocabulary))
wrd2int = {w:i for i,w in enumerate(vocabulary)}
' '.join( int2wrd[int(i)] for i in vectorizer(test_sentence) )

"""
mask_zero=True: use 0 for padding
input_dim: vocabulary_size (if mask_zero=False) or vocabulary_size + 1 (if mask_zero=True)
output_dim: dimension of the dense embedding
mask_zero: whether the input value 0 is a special "padding" value that should be masked out. 
"""
embedding_layer = Embedding( input_dim=10, output_dim=256, mask_zero=True)
inputs = [ [4, 3, 2, 1, 0, 0, 0],
           [5, 4, 3, 2, 1, 0, 0],
           [2, 1, 0, 0, 0, 0, 0] ]
embedding_layer.compute_mask( inputs )

# os.chdir(f'{os.getenv("HOME")}/Data')
text_ds = tf.data.TextLineDataset("txt/nietzsche.txt").batch(3) # creates a Dataset comprising lines from one or more text files
for lines in text_ds.take(2): 
    print(lines) # filter out empty lines

#%% Sequence Tensor: (time steps, feature dimension)

from keras.utils import timeseries_dataset_from_array

input_len = 3   # length of the input sequence used for prediction
ahead = 2       # the step ahead of (the last step of) the input sequence to predict

seq = np.arange(8)
# Targets corresponding to timesteps in data. targets[i] should be the target corresponding to the window that starts at index i
ts_dataset = timeseries_dataset_from_array(
    data = seq[:-ahead],  # the last moving window ends at seq[ len(seq)-ahead-1 ] for predicting seq[len(seq)-1], i.e., seq[-1]
    targets = seq[input_len + ahead - 1: ], # if ahead=1, the first to be predicted is seq[ seq_len ]
    sequence_length = input_len, batch_size = 2)

for inputs, targets in ts_dataset:
    print(f'inputs.shape={inputs.shape},\t targets.shape={targets.shape}')
    for i in range(inputs.shape[0]):
         print( [int(x) for x in inputs[i]], int(targets[i]))