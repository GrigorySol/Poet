from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import keras.utils as ku
import numpy as np
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.models import Sequential

tokenizer = Tokenizer()

def dataset_generation():
    f = open("data/robert_frost.txt","r")
    text = f.read()
    corpus = text.lower().split("\n")
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    max_sequence_length = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences,
                                maxlen = max_sequence_length, padding='pre'))
    predictors, label = input_sequences[:, :-1], input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)

    return predictors, label, max_sequence_length, total_words

def create_model(max_sequence_length, total_words):
    input_len = max_sequence_length - 1
    model = Sequential()
    model.add(Embedding(total_words, 10, input_length = input_len))
    model.add(LSTM(150))
    model.add(Dropout(0.1))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

def generate_text(seed_text, next_words, max_sequence_length, model):
    for j in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index==predicted:
                output_word=word
                break
        seed_text += " " + output_word

    return seed_text


X, Y, max_len, total_words = dataset_generation()

model = create_model(max_len, total_words)
model.fit(X, Y, epochs=100, verbose=1)
model.save('frost.h5')

# Loading a model
# model = create_model(max_len, total_words)
# model.load_weights("cords.h5")

text = generate_text("Sky ", 7, max_len, model)
print(text)
