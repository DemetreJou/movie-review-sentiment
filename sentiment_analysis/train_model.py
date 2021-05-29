import json
import os
import pickle
import re

import pandas as pd
from bs4 import BeautifulSoup
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

lemmatizer = WordNetLemmatizer()
import matplotlib.pyplot as plt
import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.add_pipe('sentencizer')


def clean_sentence(sentence):
    review_text = BeautifulSoup(sentence, features="html.parser").get_text()

    # remove non-alphabetic characters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)

    # tokenize the sentences
    words = word_tokenize(review_text.lower())

    # lemmatize each word to its lemma
    lemma_words = [lemmatizer.lemmatize(i) for i in words]
    return lemma_words


def clean_sentences(df):
    reviews = []
    for sent in tqdm(df['Phrase']):
        reviews.append(clean_sentence(sent))

    return reviews


if __name__ == "__main__":
    # DISABLES CUDA
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    train = pd.read_csv(os.path.join("data", "data.csv"))
    train_sentences = clean_sentences(train)

    target = train.Sentiment.values
    y_target = to_categorical(target)
    num_classes = y_target.shape[1]

    X_train, X_val, y_train, y_val = train_test_split(train_sentences, y_target, test_size=0.2, stratify=y_target)

    unique_words = set()
    len_max = 0
    for sent in tqdm(X_train):

        unique_words.update(sent)

        if (len_max < len(sent)):
            len_max = len(sent)

    tokenizer = Tokenizer(num_words=len(list(unique_words)))
    tokenizer.fit_on_texts(list(X_train))

    X_train = tokenizer.texts_to_sequences(X_train)
    X_val = tokenizer.texts_to_sequences(X_val)

    X_train = sequence.pad_sequences(X_train, maxlen=len_max)
    X_val = sequence.pad_sequences(X_val, maxlen=len_max)

    early_stopping = EarlyStopping(min_delta=0.001, mode='max', monitor='val_accuracy', patience=2)
    callback = [early_stopping]

    model = Sequential()
    model.add(Embedding(len(list(unique_words)), 300, input_length=len_max))

    model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
    model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.5, return_sequences=False))

    # TODO: can flip this flag
    complicated_model = False
    if complicated_model:
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

    else:
        model.add(Dense(num_classes, activation="sigmoid"))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.005),
        metrics=['accuracy']
    )
    model.summary()

    callbacks = []
    early_stop = False
    if early_stop:
        callbacks.append(EarlyStopping(min_delta=0.001, mode='max', monitor='val_accuracy', patience=2))

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=6,
        batch_size=256,
        verbose=1,
        callbacks=callbacks
    )

    # Create count of the number of epochs
    epoch_count = range(1, len(history.history['loss']) + 1)

    # Visualize learning curve. Here learning curve is not ideal. It should be much smoother as it decreases.
    # As mentioned before, altering different hyper parameters especially learning rate can have a positive impact
    # on accuracy and learning curve.
    plt.plot(epoch_count, history.history['loss'], 'r--')
    plt.plot(epoch_count, history.history['val_loss'], 'b-')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # TODO: comment out the saving for now just to test new models out
    save_model = False
    if save_model:
        tokenizer_json = tokenizer.to_json()
        with open(os.path.join("trained_model", "tokenizer"), 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))

        values_to_save = {
            "len_max": len_max
        }
        with open(os.path.join("trained_model", "values"), 'wb') as f:
            pickle.dump(values_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

        model.save(os.path.join("trained_model", "keras_model"))
