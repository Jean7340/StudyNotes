#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted from "Neural machine translation with a Transformer and Keras" 
https://www.tensorflow.org/text/tutorials/transformer
"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.layers import Dense, Dropout, MultiHeadAttention, Add, LayerNormalization, TextVectorization, Embedding

assert tf.__version__>= "2.18.0"
#%% Positional Embedding

class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, d_emb):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, d_emb)
        self.d_emb = d_emb
        self.pos_encoding = self.get_positional_encoding(2048, d_emb)

    def get_positional_encoding(self, seq_len, d_emb):
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(d_emb)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_emb))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.cast(angle_rads, dtype=tf.float32)

    def call(self, x):
        # Ensure input is always 2D: (batch_size, sequence_length)
        x = tf.reshape(x, [-1, tf.shape(x)[-1]])

        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_emb, tf.float32))
        return x + self.pos_encoding[tf.newaxis, :seq_len, :]


#%% Attention

class BaseAttention( layers.Layer ):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = MultiHeadAttention(**kwargs)
    self.layernorm = LayerNormalization()
    self.add = Add()
    
class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha( query=x, value=x, key=x)
    return self.layernorm( self.add([x, attn_output]) )

class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha( query=x, key=context, value=context, return_attention_scores=True)
    self.last_attn_scores = attn_scores # cache the attention scores for plotting later.
    return self.layernorm( self.add( [x, attn_output] ) )

class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha( query=x, value=x, key=x,  use_causal_mask = True)
    return self.layernorm( self.add([x, attn_output]) )

#%% Encoder

class FeedForward( keras.layers.Layer ):
  def __init__(self, d_emb, d_ff, dropout_rate=0.1):
    super().__init__()
    self.seq = keras.Sequential([
      Dense( d_ff, activation='relu' ),
      Dense( d_emb ), # project back to d_emb-dimensional space
      Dropout( dropout_rate) ])
    self.add = Add()
    self.layer_norm = LayerNormalization()
  def call(self, x):
    return self.layer_norm( self.add([x, self.seq(x)]) ) 

class EncoderLayer(layers.Layer):
  def __init__(self, * , d_emb, n_heads, d_ff, dropout_rate=0.1): # parameters after * or *identifier are keyword-only parameters and may only be passed used keyword arguments
    super().__init__()
    self.attention = GlobalSelfAttention( num_heads=n_heads,  key_dim=d_emb,  dropout=dropout_rate )
    self.ffn = FeedForward(d_emb, d_ff)
  def call(self, x):
    x = self.attention(x)
    return self.ffn(x)

class Encoder(layers.Layer):
  def __init__(self, *, n_layers, d_emb, n_heads, d_ff, dropout_rate=0.1):
    super().__init__()
    self.d_emb = d_emb
    self.n_layers = n_layers
    self.enc_layers = [ EncoderLayer(d_emb=d_emb, n_heads=n_heads, d_ff=d_ff, dropout_rate=dropout_rate) for _ in range(n_layers) ]
    self.dropout = Dropout(dropout_rate)
  
  def call(self, x): # x is token-IDs shape: (batch, seq_len)
    x = self.dropout( x )
    for i in range(self.n_layers):
      x = self.enc_layers[i](x)
    return x  # Shape: (batch_size, seq_len, d_emb)

#%% Decoder

class DecoderLayer(layers.Layer):
  def __init__(self, *, d_emb, n_heads, d_ff, dropout_rate=0.1):
    super().__init__()
    self.causal_self_attention = CausalSelfAttention( num_heads=n_heads, key_dim=d_emb, dropout=dropout_rate )
    self.cross_attention = CrossAttention( num_heads=n_heads, key_dim=d_emb, dropout=dropout_rate )
    self.ffn = FeedForward( d_emb, d_ff )
  def call(self, x, context):
    x = self.causal_self_attention(x)
    x = self.cross_attention( x, context )
    self.last_attn_scores = self.cross_attention.last_attn_scores # cache the last attention scores for plotting later
    return self.ffn(x)  # Shape: (batch_size, seq_len, d_emb)

class Decoder(layers.Layer):
  def __init__(self, *, n_layers, d_emb, n_heads, d_ff, dropout_rate=0.1):
    super().__init__()
    self.d_emb = d_emb
    self.n_layers = n_layers
    self.dropout = Dropout(dropout_rate)
    self.dec_layers = [ DecoderLayer(d_emb=d_emb, n_heads=n_heads, d_ff=d_ff, dropout_rate=dropout_rate) for _ in range(n_layers) ]
    self.last_attn_scores = None

  def call(self, x, context): # x is of shape (batch, tgt_seq_len); context is of shape (batch_size, context_len, d_emb)
    x = self.dropout(x)
    for i in range(self.n_layers):
      x  = self.dec_layers[i](x, context)
    self.last_attn_scores = self.dec_layers[-1].last_attn_scores    
    return x # shape: (batch_size, tgt_seq_len, d_emb)

#%% Transformer

class Transformer(keras.Model):
    def __init__(self, *, n_layers, d_emb, n_heads, d_ff, src_vocab_size, tgt_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.src_pos_embedding = PositionalEmbedding(vocab_size=src_vocab_size, d_emb=d_emb)
        self.tgt_pos_embedding = PositionalEmbedding(vocab_size=tgt_vocab_size, d_emb=d_emb)
        self.encoder = Encoder(n_layers=n_layers, d_emb=d_emb, n_heads=n_heads, d_ff=d_ff, dropout_rate=dropout_rate)
        self.decoder = Decoder(n_layers=n_layers, d_emb=d_emb, n_heads=n_heads, d_ff=d_ff, dropout_rate=dropout_rate)
        self.final_layer = Dense(tgt_vocab_size)

    def call(self, inputs, training=False):  # <- include training arg
        src = inputs[0]
        tgt = inputs[1]

        src_emb = self.src_pos_embedding(src)
        tgt_emb = self.tgt_pos_embedding(tgt)

        context = self.encoder(src_emb)
        x = self.decoder(tgt_emb, context)

        logits = self.final_layer(x)

        try:
            del logits._keras_mask  # clean mask if present
        except AttributeError:
            pass

        return logits


#%% Sanity Check
if __name__ == "__main__":    
    vocab_size = 1000
    seq_len = 10
    d_emb = 512
        
    en = ['good morning', 'how are you']
    sp = ['buen dia', 'Como estas']

    src_vectorizer = TextVectorization(  max_tokens=vocab_size, output_mode="int", output_sequence_length=seq_len )
    tgt_vectorizer = TextVectorization(  max_tokens=vocab_size, output_mode="int", output_sequence_length=seq_len )
    src_vectorizer.adapt( en )
    tgt_vectorizer.adapt( sp )

    seq_en = src_vectorizer( en )
    seq_sp = tgt_vectorizer( sp )
    emb_en = PositionalEmbedding( vocab_size=vocab_size, d_emb=d_emb )( seq_en )
    emb_sp = PositionalEmbedding( vocab_size=vocab_size, d_emb=d_emb )( seq_en )
    
    n_heads = 8    
    gsa = GlobalSelfAttention( num_heads=n_heads, key_dim=d_emb )
    ca  = CrossAttention( num_heads=n_heads, key_dim=d_emb )
    csa = CausalSelfAttention( num_heads=n_heads, key_dim=d_emb )
    gsa( emb_en ).shape
    ca( emb_sp, emb_en ).shape
    csa( emb_sp ).shape # ensure tf. __version__ > = 2.10.0
 
    d_ff = 2048
    encoder_layer = EncoderLayer( d_emb=d_emb, n_heads=n_heads, d_ff=d_ff )
    decoder_layer = DecoderLayer( d_emb=d_emb, n_heads=n_heads, d_ff=d_ff )
    encoder_layer( emb_en ).shape
    decoder_layer(x=emb_sp, context=emb_en).shape    

    n_layers = 6    
    encoder = Encoder( n_layers=n_layers, d_emb=d_emb, n_heads=n_heads, d_ff=d_ff )
    decoder = Decoder( n_layers=n_layers, d_emb=d_emb, n_heads=n_heads, d_ff=d_ff )
    encoder( emb_en, training=False).shape   
    decoder( x=emb_sp, context=emb_en ).shape
else:
    print(f'Transformer imported from local file "{__name__}.py"')
