#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
  
import keras
from keras.layers import Input, Dense, Reshape, Conv1D, MaxPooling1D, GlobalAveragePooling1D, SimpleRNN, LSTM, GRU, Dropout, Bidirectional

import matplotlib.pyplot as plt
def plotEpoch( history ):    
    epochs = range(len(history['mae']) )
    plt.plot(epochs, history['loss'], 'bo', label='Training loss')
    plt.plot(epochs, history['val_loss'], 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show() 

# os.chdir(f'{os.getenv("HOME")}/Data')
#%% Data Preparation

with open("csv/jena_climate_2009_2016.csv") as f: # https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip
    data = f.read()
lines = data.split("\n")
header = lines[0].split(",") # 14 weather-related values
lines = lines[1:] # weather records every 10 min
n = len(lines)

temperature = np.zeros( (n,) )
raw_data = np.zeros( (n, len(header)-1) ) # the 1st column is datetime
for i, line in enumerate(lines):
    values = [ float(x) for x in line.split(",")[1:] ] # discard the 1st column (i.e. datetime)
    temperature[i] = values[1] # the 3rd column in the original csv
    raw_data[i, :] = values[:]

plt.plot(range(n), temperature)
plt.plot(range(1440), temperature[:1440]) # first 10 days: 24*6=144

num_train_samples, num_val_samples = int( 0.5 * n ), int( 0.25 * n )
num_test_samples = n - num_train_samples - num_val_samples
mean, std = raw_data[:num_train_samples].mean(axis=0), raw_data[:num_train_samples].std(axis=0) # calculate the mean and std of each column, only using training data
raw_data = (raw_data - mean) /std

from keras.utils import timeseries_dataset_from_array
sampling_rate = 6 # sample one record per hour
seq_len = 120 # use 5 days (5*24)  of record history
ahead = 24 # predict 24 hours ahead
shift_y = sampling_rate * (seq_len + ahead - 1)
shift_x = sampling_rate * ahead
batch_size = 256

train_dataset = timeseries_dataset_from_array( raw_data[:-shift_x], targets=temperature[shift_y:],
    sampling_rate=sampling_rate, sequence_length=seq_len, batch_size=batch_size, start_index=0, end_index=num_train_samples, shuffle=True)

val_dataset = timeseries_dataset_from_array( raw_data[:-shift_x], targets=temperature[shift_y:],
    sampling_rate=sampling_rate, sequence_length=seq_len, batch_size=batch_size, start_index=num_train_samples, end_index=num_train_samples + num_val_samples, shuffle=True)

test_dataset = timeseries_dataset_from_array( raw_data[:-shift_x], targets=temperature[shift_y:],
    sampling_rate=sampling_rate, sequence_length=seq_len, batch_size=batch_size, start_index=num_train_samples + num_val_samples, shuffle=True)

for inputs, targets in train_dataset.take(1):
    print(f'inputs.shape={inputs.shape},\t targets.shape={targets.shape}')
    print(f'input={inputs[0]}, \t target={targets[0]}')
        
#%% A naive benchmark, and two inadequate architecture

total_abs_err, samples_seen = 0., 0
for samples, targets in test_dataset:
    preds = samples[:, -1, 1] * std[1] + mean[1] # use the last temperature (2nd column) which is 24 hours before the target
    total_abs_err += np.sum(np.abs(preds - targets))
    samples_seen += samples.shape[0]
print(f"Test MAE: {total_abs_err / samples_seen:.2f}") # 2.62

inputs = Input( shape=(seq_len, raw_data.shape[-1]) ) # process sequences of fixed length --- recommended for performance optimization. To process sequences of arbitrary length, Input( shape=(None,    raw_data.shape[-1]) )
x = Reshape((-1,))(inputs) # Flatten() bug: only one input size may be -1, not both 0 and 1
x = Dense(16, activation="relu")(x)
outputs = Dense(1)(x) # the lack of activation function is typical for a regression problem
model = keras.Model(inputs, outputs)
model.compile( optimizer="rmsprop", loss="mse", metrics=["mae"] )
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)
plotEpoch( history.history )
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}") # 2.74


#%% Conv1D

x = Conv1D(8, kernel_size=24, activation="relu")(inputs)
x = MaxPooling1D(2)(x)
x = Conv1D(8, kernel_size=12, activation="relu")(x)
x = MaxPooling1D(2)(x)
x = Conv1D(8, 6, activation="relu")(x)
x = GlobalAveragePooling1D()(x)
outputs = Dense(1)(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)
plotEpoch( history.history )
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}") # 3.06

#%% SimpleRNN

x = SimpleRNN(16)(inputs) # default: return_sequence=False
outputs = Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary()
model.compile( optimizer="rmsprop", loss="mse", metrics=["mae"] )
h1 = model.fit( train_dataset, epochs=10, validation_data=val_dataset )
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}") # 2.51

#%% LSTM

x = LSTM(16)(inputs)
outputs = Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary()

model.compile( optimizer="rmsprop", loss="mse", metrics=["mae"] )
callbacks = [ keras.callbacks.ModelCheckpoint("models/jena_lstm.keras", save_best_only=True) ]
h1 = model.fit( train_dataset, epochs=10, validation_data=val_dataset, callbacks=callbacks )
plotEpoch( h1.history )
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}") # 2.59

#%% GRU

x = GRU(32, recurrent_dropout=0.25)(inputs) # using dropout as regularization, we can afford larger number of units.
x = Dropout(0.5)(x)
outputs = Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary()

model.compile( optimizer="rmsprop", loss="mse", metrics=["mae"] )
callbacks = [ keras.callbacks.ModelCheckpoint("models/jena_gru.keras", save_best_only=True) ]
h2 = model.fit( train_dataset, epochs=10, validation_data=val_dataset, callbacks=callbacks ) # Networks regularized with dropout take longer to fully converge. Use more epochs.
plotEpoch( h2.history )
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}") # 2.42

#%% Stacking Recurrent Layers

x = GRU(32, recurrent_dropout=0.5, return_sequences=True)(inputs)
x = GRU(32, recurrent_dropout=0.5)(x)
x = Dropout(0.5)(x)
outputs = Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
callbacks = [ keras.callbacks.ModelCheckpoint("models/jena_stacked_gru.keras", save_best_only=True) ]
h3 = model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=callbacks)
plotEpoch( h3.history )
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}") # 2.44

#%% Bidirectional

x = Bidirectional( LSTM(16) )(inputs)
outputs = Dense(1)(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
callbacks = [ keras.callbacks.ModelCheckpoint("models/jena_bidirectional.keras", save_best_only=True) ]
h4 = model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=callbacks)
plotEpoch( h4.history )
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")  # 2.56