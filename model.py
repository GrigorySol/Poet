from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
import numpy as np

def create_model(predictors, label, max_sequence_length, total_words):
    input_len = max_sequence_length - 1
    model = Sequential()
    model.add(Embedding(total_words, 10, input_length = input_len))
    model.add(LSTM(150))
    model.add(Dropout(0.1))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(predictors, label, epochs=100, verbose=100)

    return model
