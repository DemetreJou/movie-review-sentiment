import json
import pickle

import numpy as np
import keras
from keras_preprocessing.text import tokenizer_from_json
from keras.preprocessing import sequence
import os
from sentiment_analysis.train_model import clean_sentence

model = keras.models.load_model(os.path.join("trained_model", "keras_model"))
with open(os.path.join( "trained_model", "tokenizer")) as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

with open(os.path.join("..", "sentiment_analysis", "trained_model", "values"), 'rb') as handle:
    values = pickle.load(handle)


def get_sentiment(phrase):
    phrase = clean_sentence(phrase)
    seq = tokenizer.texts_to_sequences(phrase)
    X_new = sequence.pad_sequences(seq, maxlen=values["len_max"])
    y_preds = np.argmax(model.predict(X_new), axis=-1)
    return y_preds


if __name__ == "__main__":
    phrases = [
        "poetically states at one point in this movie that we `` do n't care about the truth",
        "poetically",
        "Must see summer blockbuster",
        "A comedy-drama of nearly epic proportions rooted in a sincere performance by the title character undergoing midlife crisis"
    ]
    for phrase in phrases:
        result = get_sentiment(phrase)
        print(f"{phrase} : {result}")


