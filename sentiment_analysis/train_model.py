import os
import pickle
import re
from enum import IntEnum

import keras
import numpy as np
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


# not sure if this is needed or if a dict is better
class sentiment(IntEnum):
    NEGATIVE = 0
    SOMEWHAT_NEGATIVE = 1
    NEUTRAL = 2
    SOMEWHAT_POSITIVE = 3
    POSITIVE = 4


class SentimentModel:
    save_base_path: str
    data_base_path: str
    VOCAB_SIZE: int
    MAX_LEN: int

    # TODO: add these type hints
    # model: tf.keras.Model
    # model_history:

    def __init__(
            self,
            *,
            vocab_size: int = 50000,
            max_len: int = 200,
            save_base_path: str = "trained_model",
            data_base_path: str = "data",
            load_pretrained: bool = False

    ):
        self.VOCAB_SIZE = vocab_size
        self.MAX_LEN = max_len
        self.save_base_path = save_base_path
        self.data_base_path = data_base_path

        # Not sure if the "else" case is required here
        if load_pretrained:
            self.model = self.load_model()
        else:
            self.model = None
        self.model_history = None

    def load_model(self):
        return keras.models.load_model(os.path.join(self.save_base_path, "keras_model"))

    def save_model(self):
        values_to_save = {
            "max_len": self.MAX_LEN,
            "vocab_size": self.VOCAB_SIZE
        }
        with open(os.path.join(self.save_base_path, "values"), 'wb') as f:
            pickle.dump(values_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.model.save(os.path.join(self.save_base_path, "keras_model"))

    def get_sentiment(self, sentence: str) -> sentiment:
        if self.model is None:
            raise Exception("Model does not exist. Try loading a previously saved model or training this model")
        sentence = self.clean_sentence(sentence)
        sentence = [one_hot(d, self.VOCAB_SIZE) for d in sentence]
        sentence = keras.preprocessing.sequence.pad_sequences([sentence], maxlen=self.MAX_LEN)
        prediction = np.argmax(self.model.predict(sentence), axis=-1)
        return sentiment(prediction)

    def clean_sentence(self, sentence):
        review_text = BeautifulSoup(sentence, features="html.parser").get_text()

        # remove non-alphabetic characters
        review_text = re.sub("[^a-zA-Z]", " ", review_text)

        # tokenize the sentences
        words = word_tokenize(review_text.lower())

        # lemmatize each word to its lemma
        return ' '.join([lemmatizer.lemmatize(i) for i in words])

    def clean_sentences(self, df):
        reviews = []
        for sent in tqdm(df['Phrase']):
            reviews.append(self.clean_sentence(sent))

        return reviews

    def text_cleaning(self, text):
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

    def preprocess_list(self, lst):
        lst = list(map(self.clean_sentence, lst))
        lst = [one_hot(x, self.VOCAB_SIZE) for x in lst]
        lst = keras.preprocessing.sequence.pad_sequences(lst, maxlen=self.MAX_LEN)
        return lst

    def load_and_preprocess(self):
        train_data = pd.read_csv(os.path.join("data", "train.tsv"), sep='\t')
        test_data = pd.read_csv(os.path.join("data", "test.tsv"), sep='\t')

        train_data = train_data.drop(['PhraseId', 'SentenceId'], axis=1)
        test_data = test_data.drop(['PhraseId', 'SentenceId'], axis=1)
        # TODO: refactor out this total_docs stuff, just preprocess train, test seperately
        # train_data['flag'] = 'TRAIN'
        # test_data['flag'] = 'TEST'
        # total_docs = pd.concat([train_data, test_data], axis=0, ignore_index=True)
        # total_docs['Phrase'] = self.preprocess_list(total_docs['Phrase'].tolist())
        # train_data = total_docs[total_docs['flag'] == 'TRAIN']
        # test_data = total_docs[total_docs['flag'] == 'TEST']
        # x_train = train_data['Phrase']
        # y_train = train_data['Sentiment']
        # x_val = test_data['Phrase']
        # y_val = test_data['Sentiment']
        x_train = self.preprocess_list(train_data['Phrase'].tolist())
        y_train = train_data['Sentiment']
        x_val = self.preprocess_list(test_data['Phrase'].tolist())
        y_val = test_data['Sentiment']

        return x_train, x_val, y_train, y_val

    def generate_model(self):
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
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def train_model(self):
        self.model = self.generate_model()
        x_train, x_val, y_train, y_val = self.load_and_preprocess()
        self.model_history = self.model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))

    def visualize_loss(self):
        if self.model_history is None:
            raise Exception("No model history found. Must train model")
        epoch_count = range(1, len(self.model_history.history['loss']) + 1)
        plt.plot(epoch_count, self.model_history.history['loss'], 'r--')
        plt.plot(epoch_count, self.model_history.history['val_loss'], 'b-')
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()


if __name__ == "__main__":
    # DISABLES CUDA
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    model = SentimentModel()
    model.train_model()
    model.visualize_loss()
    model.save_model()
