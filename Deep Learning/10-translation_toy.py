#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.layers import Input, Embedding, Dense, Dropout, TextVectorization, GRU, Bidirectional

import os
# os.chdir(f'{os.getenv("HOME")}/analytics/lib') # place this before data preparation
from transformer import Transformer

import string
import re
import random

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")
def custom_standardization(input_string):
    return tf.strings.regex_replace( tf.strings.lower(input_string), f"[{re.escape(strip_chars)}]", "" )

# os.chdir(f'{os.getenv("HOME")}/Data')
text_file = "spa-eng/spa.txt" # http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip
with open(text_file) as f:
    lines = f.read().split("\n")[:-1]

#%% Dataset preparation for English-to-Spanish translation
    
text_pairs = []
for line in lines:
    english, spanish = line.split("\t")
    spanish = "[start] " + spanish + " [end]" # prepend an SOS token and append an EOS token to the target sequence
    text_pairs.append( (english, spanish) )

random.shuffle( text_pairs )
num_val = int(0.15 * len(text_pairs))
num_train = len(text_pairs) - 2 * num_val
train_pairs, val_pairs, test_pairs = text_pairs[:num_train], text_pairs[num_train:num_train+num_val], text_pairs[num_train+num_val:]

vocab_size, seq_length = 15000, 20

source_vectorization = TextVectorization(  max_tokens=vocab_size, output_mode="int", 
                                           output_sequence_length=seq_length )
source_vectorization.adapt( [pair[0] for pair in train_pairs] )

# target sequence is one token longer than the source sequence since we'll construct spa[:, :-1]) and spa[:, 1:]
target_vectorization = TextVectorization(  max_tokens=vocab_size, output_mode="int", 
                                           standardize=custom_standardization,
                                           output_sequence_length=seq_length + 1)
target_vectorization.adapt( [pair[1] for pair in train_pairs] )

def format_dataset(eng, spa):
    eng = source_vectorization( eng )
    spa = target_vectorization( spa )
    return ( (eng, spa[:, :-1]), spa[:, 1:]) # target is one step ahead

batch_size = 64
def make_dataset(pairs):
    eng_texts, spa_texts = zip( *pairs ) # unzip the sequence of (en,sp) pairs 
    eng_texts, spa_texts = list(eng_texts), list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices( (eng_texts, spa_texts) ).batch( batch_size ).map( format_dataset, num_parallel_calls=4 )
    return dataset.shuffle(2048).prefetch(16).cache() # use in-memory catching to speed up preprocessing

train_ds = make_dataset( train_pairs )
val_ds = make_dataset( val_pairs )

#%% Loss, Accuracy, and Optimizer

def masked_loss(label, pred):
  loss = keras.losses.SparseCategoricalCrossentropy( from_logits=True, reduction='none' )(label, pred)
  mask = tf.cast(label != 0, dtype=loss.dtype)
  loss *= mask
  return tf.reduce_sum(loss)/tf.reduce_sum(mask)

def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  mask = label != 0  
  match = (label == pred) & mask
  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)

#%% RNN

embed_dim = 256
latent_dim = 1024
source = Input(shape=(None,), dtype="int64", name="english")
x = Embedding(vocab_size, embed_dim, mask_zero=True)( source ) # padding mark is on 
encoded_source = Bidirectional( layers.GRU(latent_dim), merge_mode="sum" )(x)    
target_prev_steps = Input( shape=(None,), dtype="int64", name="spanish" ) # spa[:, :-1]
x = Embedding(vocab_size, embed_dim, mask_zero=True)(target_prev_steps) # padding mark is on 
x = GRU( latent_dim, return_sequences=True )( x, initial_state=encoded_source )
x = Dropout(0.5)(x)
target_next_steps = Dense( vocab_size, activation="softmax" )(x) # output a distribution at EACH step for the next token
rnn = keras.Model( [source, target_prev_steps], target_next_steps ) # target_next_steps is trained to match spa[:, 1:]
rnn.compile( loss=masked_loss, optimizer="rmsprop", metrics=[masked_accuracy] )
rnn.fit( train_ds, epochs=10, validation_data=val_ds ) # val_masked_accuracy = 60% after 15 epochs

#%% Transformer

class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()
    self.d_model = tf.cast(d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(128)
optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
  
model = Transformer( n_layers=4, d_emb=128, n_heads=8, d_ff=512, dropout_rate=0.1, src_vocab_size=vocab_size, tgt_vocab_size=vocab_size )
model.compile( loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy] )
model.fit( train_ds, epochs=10, validation_data=val_ds ) # val_masked_accuracy = 69% after 10 epochs

#%% Translate

spa_vocab = target_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab)) # a dict to convert token index prediction to string token

def decode_sequence( input_sentence ):
    tokenized_input_sentence = source_vectorization( [input_sentence] )
    decoded_sentence = "[start]" # seed token
    for i in range( 20 ): # 20 tokens at most for the decoded sentence
        tokenized_target_sentence = target_vectorization( [decoded_sentence] ) # [:, :-1]
        next_token_predictions = model.predict( [tokenized_input_sentence, tokenized_target_sentence] )
        sampled_token_index = np.argmax( next_token_predictions[0, i, :] )
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence

test_eng_texts = [ pair[0] for pair in test_pairs ]
for _ in range(5):
    input_sentence = random.choice(test_eng_texts)
    print("-")
    print(input_sentence)
    print(decode_sequence(input_sentence))
