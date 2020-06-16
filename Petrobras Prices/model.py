# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:00:38 2020

@author: iago
"""

from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential


def build_model(output, hidden_units, shape,
                optimizer, loss):
    model = Sequential()
    
    model.add(SimpleRNN(hidden_units, return_sequences = True, input_shape = shape))
    
    model.add(Dropout(0.1))
    
    model.add(SimpleRNN(hidden_units, return_sequences = True))
    
    model.add(Dropout(0.1))
    
    model.add(SimpleRNN(hidden_units, return_sequences = True))
    
    model.add(Dropout(0.1))
    
    model.add(SimpleRNN(hidden_units, return_sequences = True))
    
    model.add(Dropout(0.1))
    
    model.add(Dense(output))
    
    
    model.compile(optimizer = optimizer, loss = loss)
    
    return model