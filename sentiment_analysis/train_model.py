import numpy as np
import pandas as pd
import os

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

lemmatizer = WordNetLemmatizer()
import re

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy
from bs4 import BeautifulSoup
from keras.optimizers import Adam
from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import matplotlib.pyplot as plt
import html
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.add_pipe('sentencizer')

from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
pool = Pool(cpu_count())


def clean_sentences(df, steps=range(1, 6)):
    reviews = []

    for sent in tqdm(df['Phrase']):
        # modComm = sent
        # # treat as blank string
        # if modComm == "[deleted]" or modComm == "[removed]":
        #     modComm = ""
        #
        # if 1 in steps:
        #     modComm = re.sub(r"\n{1,}", " ", modComm)
        #     modComm = re.sub(r"\t{1,}", " ", modComm)
        #     modComm = re.sub(r"\r{1,}", " ", modComm)
        #     modComm = re.sub(r"\r\n{1,}", " ", modComm)
        #
        # if 2 in steps:  # unescape html
        #     modComm = html.unescape(modComm)
        #
        # if 3 in steps:  # remove URLs
        #     modComm = re.sub(r"(http|www)\S+", "", modComm)
        #
        # if 4 in steps:  # remove duplicate spaces.
        #     modComm = re.sub(r" +", " ", modComm)
        #
        # if 5 in steps:
        #     utt = nlp(modComm)
        #     modComm = ""
        #
        #     for sentence in utt.sents:
        #         for token in sentence:
        #             if token.lemma_.startswith("-") and not token.text.startswith("-"):
        #                 if token.text.isupper():
        #                     modComm += f"{token.text}/{token.tag_} "
        #                 else:
        #                     modComm += f"{token.text.lower()}/{token.tag_} "
        #
        #             elif token.text.isupper():
        #                 modComm += f"{token.lemma_.upper()}/{token.tag_} "
        #             else:
        #                 modComm += f"{token.lemma_.lower()}/{token.tag_} "
        #
        #         # remove space at end of sentence
        #         if modComm != "":
        #             modComm = modComm[:-1]
        #         modComm += "\n"

        # remove html content
        review_text = BeautifulSoup(sent).get_text()

        # remove non-alphabetic characters
        review_text = re.sub("[^a-zA-Z]", " ", review_text)

        # tokenize the sentences
        words = word_tokenize(review_text.lower())

        # lemmatize each word to its lemma
        lemma_words = [lemmatizer.lemmatize(i) for i in words]

        reviews.append(lemma_words)

        #reviews.append(modComm)

    return (reviews)


def clean_sentence(sent, steps=range(1, 6)):
    modComm = sent
    # treat as blank string
    if modComm == "[deleted]" or modComm == "[removed]":
        modComm = ""

    if 1 in steps:
        modComm = re.sub(r"\n{1,}", " ", modComm)
        modComm = re.sub(r"\t{1,}", " ", modComm)
        modComm = re.sub(r"\r{1,}", " ", modComm)
        modComm = re.sub(r"\r\n{1,}", " ", modComm)

    if 2 in steps:  # unescape html
        modComm = html.unescape(modComm)

    if 3 in steps:  # remove URLs
        modComm = re.sub(r"(http|www)\S+", "", modComm)

    if 4 in steps:  # remove duplicate spaces.
        modComm = re.sub(r" +", " ", modComm)

    if 5 in steps:
        utt = nlp(modComm)
        modComm = ""

        for sentence in utt.sents:
            for token in sentence:
                if token.lemma_.startswith("-") and not token.text.startswith("-"):
                    if token.text.isupper():
                        modComm += f"{token.text}/{token.tag_} "
                    else:
                        modComm += f"{token.text.lower()}/{token.tag_} "

                elif token.text.isupper():
                    modComm += f"{token.lemma_.upper()}/{token.tag_} "
                else:
                    modComm += f"{token.lemma_.lower()}/{token.tag_} "

            # remove space at end of sentence
            if modComm != "":
                modComm = modComm[:-1]
            modComm += "\n"

    return modComm


if __name__ == "__main__":
    train = pd.read_csv(os.path.join("data", "data.csv"))
    train_sentences = clean_sentences(train)

    # train_sentences = pool.map(clean_sentence, train['Phrase'])
    # conver to one hot encoding for targets
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

    early_stopping = EarlyStopping(min_delta=0.001, mode='max', monitor='val_acc', patience=2)
    callback = [early_stopping]

    model = Sequential()
    model.add(Embedding(len(list(unique_words)), 300, input_length=len_max))
    model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
    model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.5, return_sequences=False))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.005), metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=6, batch_size=256, verbose=1, callbacks=callback)


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

    model.save(os.path.join("trained_model"))


    # seq = tokenizer.texts_to_sequences(ns)
    # X_new = sequence.pad_sequences(seq, maxlen=len_max)
    # y_preds = model.predict_classes(X_new)
    pass
