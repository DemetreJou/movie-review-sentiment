# issues with lazy loading causing false positives
# pylint: disable=no-name-in-module
import os
import pickle
import re
from enum import IntEnum
from typing import List

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import spacy
import tensorflow.keras as keras
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.add_pipe('sentencizer')


# not sure if this is needed or if a dict is better
class Sentiment(IntEnum):
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
    # sentinment_model: tf.keras.Model
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
        # issues with these constants
        # pylint: disable=invalid-name
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
        return keras.models.load_model(os.path.join('.', 'sentiment_analysis', self.save_base_path, "keras_model"))

    def save_model(self):
        values_to_save = {
            "max_len": self.MAX_LEN,
            "vocab_size": self.VOCAB_SIZE
        }
        with open(os.path.join(self.save_base_path, "values"), 'wb') as file:
            pickle.dump(values_to_save, file, protocol=pickle.HIGHEST_PROTOCOL)

        self.model.save(os.path.join('.', 'sentiment_analysis', self.save_base_path, "keras_model"))

    def get_sentiment(self, sentence: str) -> Sentiment:
        if self.model is None:
            raise Exception("Model does not exist. Try loading a previously saved sentinment_model or training this sentinment_model")
        sentence = self._preprocess_list([sentence])
        prediction = np.argmax(self.model.predict(sentence), axis=-1)
        return Sentiment(prediction)

    @staticmethod
    def _clean_sentence(sentence: str):
        review_text = BeautifulSoup(sentence, features="html.parser").get_text()

        # remove non-alphabetic characters
        review_text = re.sub("[^a-zA-Z]", " ", review_text)

        # tokenize the sentences
        words = word_tokenize(review_text.lower())

        # lemmatize each word to its lemma
        return ' '.join([lemmatizer.lemmatize(i) for i in words])

    @staticmethod
    def text_cleaning(text):
        forbidden_words = set(stopwords.words('english'))
        if text:
            text = ' '.join(text.split('.'))
            text = re.sub(r'\/', ' ', text)
            text = re.sub(r'\\', ' ', text)
            text = re.sub(r'((http)\S+)', '', text)
            text = re.sub(r'\s+', ' ', re.sub('[^A-Za-z]', ' ', text.strip().lower())).strip()
            text = re.sub(r'\W+', ' ', text.strip().lower()).strip()
            text = [word for word in text.split() if word not in forbidden_words]
            return text
        return []

    def _preprocess_list(self, lst: List[str]):
        lst = list(map(self._clean_sentence, lst))
        lst = [one_hot(x, self.VOCAB_SIZE) for x in lst]
        lst = keras.preprocessing.sequence.pad_sequences(lst, maxlen=self.MAX_LEN)
        return lst

    def load_and_preprocess_train_data(self):
        train_data = pd.read_csv(os.path.join("data", "train.tsv"), sep='\t')
        train_data = train_data.drop(['PhraseId', 'SentenceId'], axis=1)
        x_train = self._preprocess_list(train_data['Phrase'].tolist())
        y_train = train_data['Sentiment']
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train)

        return x_train, x_val, y_train, y_val

    @staticmethod
    def generate_model():
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
        x_train, x_val, y_train, y_val = self.load_and_preprocess_train_data()
        self.model_history = self.model.fit(x_train, y_train, batch_size=32, epochs=3, validation_data=(x_val, y_val))

    def visualize_loss(self):
        if self.model_history is None:
            raise Exception("No sentinment_model history found. Must train sentinment_model")
        epoch_count = range(1, len(self.model_history.history['loss']) + 1)
        plt.plot(epoch_count, self.model_history.history['loss'], 'r--')
        plt.plot(epoch_count, self.model_history.history['val_loss'], 'b-')
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def make_predictions(self):
        """
        Loads test file
        Preprocess through usual pipeline
        Predict classes
        Save to file in format ready to submit to kaggle
        """
        test_data = pd.read_csv(os.path.join("data", "test.tsv"), sep='\t')
        x_test = self._preprocess_list(test_data['Phrase'].tolist())
        predictions = np.argmax(self.model.predict(x_test), axis=-1)
        predictions = pd.Series(predictions)
        submission = test_data['PhraseId']
        submission = submission.to_frame()
        submission['Sentiment'] = predictions

        # drop index column before saving
        submission.to_csv(os.path.join("data", "submission.csv"), index=False)


if __name__ == "__main__":
    # DISABLES CUDA
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    TRAIN_MODEL = True

    if TRAIN_MODEL:
        sentinment_model = SentimentModel()
        sentinment_model.train_model()
        sentinment_model.visualize_loss()
        sentinment_model.save_model()

    else:
        sentinment_model = SentimentModel(load_pretrained=True)
        sentinment_model.make_predictions()
