import os
import pickle
import re

import keras
import pandas as pd
from bs4 import BeautifulSoup
from keras.layers import Dense, Embedding, LSTM, Bidirectional
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

lemmatizer = WordNetLemmatizer()
import matplotlib.pyplot as plt
import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.add_pipe('sentencizer')
VOCAB_SIZE = 50000
MAX_LEN = 200


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


def text_cleaning(text):
    forbidden_words = set(stopwords.words('english'))
    if text:
        text = ' '.join(text.split('.'))
        text = re.sub('\/', ' ', text)
        text = re.sub(r'\\', ' ', text)
        text = re.sub(r'((http)\S+)', '', text)
        text = re.sub(r'\s+', ' ', re.sub('[^A-Za-z]', ' ', text.strip().lower())).strip()
        text = re.sub(r'\W+', ' ', text.strip().lower()).strip()
        text = [word for word in text.split() if word not in forbidden_words]
        return text
    return []


def load_and_process():
    train_data = pd.read_csv(os.path.join("data", "train.tsv"), sep='\t')
    test_data = pd.read_csv(os.path.join("data", "test.tsv"), sep='\t')

    train_data = train_data.drop(['PhraseId', 'SentenceId'], axis=1)
    test_data = test_data.drop(['PhraseId', 'SentenceId'], axis=1)
    train_data['flag'] = 'TRAIN'
    test_data['flag'] = 'TEST'
    total_docs = pd.concat([train_data, test_data], axis=0, ignore_index=True)
    # total_docs['Phrase'] = total_docs['Phrase'].apply(lambda x: ' '.join(text_cleaning(x)))
    total_docs['Phrase'] = total_docs['Phrase'].apply(lambda x: ' '.join(clean_sentence(x)))
    phrases = total_docs['Phrase'].tolist()
    encoded_phrases = [one_hot(d, VOCAB_SIZE) for d in phrases]
    total_docs['Phrase'] = encoded_phrases
    train_data = total_docs[total_docs['flag'] == 'TRAIN']
    test_data = total_docs[total_docs['flag'] == 'TEST']
    x_train = train_data['Phrase']
    y_train = train_data['Sentiment']
    x_val = test_data['Phrase']
    y_val = test_data['Sentiment']

    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)
    x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=MAX_LEN)
    return x_train, x_val, y_train, y_val


if __name__ == "__main__":
    # DISABLES CUDA
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    x_train, x_val, y_train, y_val = load_and_process()

    model = Sequential()
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    model.add(inputs)
    model.add(Embedding(50000, 128))
    # Add 2 bidirectional LSTMs
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(64)))
    # Add a classifier
    model.add(Dense(5, activation="sigmoid"))
    # model = keras.Model(inputs, outputs)
    model.summary()

    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    # TODO: the visualization shows that validation loss keeps decreasing from epoch 1 to 5, try training with more epochs
    model_history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_val, y_val))

    # Visualize loss/accuracy over each epoch
    epoch_count = range(1, len(model_history.history['loss']) + 1)
    plt.plot(epoch_count, model_history.history['loss'], 'r--')
    plt.plot(epoch_count, model_history.history['val_loss'], 'b-')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # TODO: comment out the saving for now just to test new models out
    save_model = False
    if save_model:
        values_to_save = {
            "max_len": MAX_LEN,
            "vocab_size": VOCAB_SIZE
        }
        with open(os.path.join("trained_model", "values"), 'wb') as f:
            pickle.dump(values_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

        model.save(os.path.join("trained_model", "keras_model"))
