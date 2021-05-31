import io
import json
import pickle
import time

import flask
import numpy as np
from flask import Flask
from flask_cors import cross_origin
from flask import request
from . import routes
import keras
from keras_preprocessing.text import tokenizer_from_json
from keras.preprocessing import sequence
import os

model = keras.models.load_model(os.path.join("..", "sentiment_analysis", "trained_model", "keras_model"))
with open(os.path.join("..", "sentiment_analysis", "trained_model", "tokenizer")) as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

with open(os.path.join("..", "sentiment_analysis", "trained_model", "values"), 'rb') as handle:
    values = pickle.load(handle)


@routes.route('/api/v1/get_sentiment', methods=['GET'])
@cross_origin()
def get_sentiment():
    phrase = request.args.get("phrase")
    seq = tokenizer.texts_to_sequences(phrase)
    X_new = sequence.pad_sequences(seq, maxlen=values["len_max"])
    y_preds = np.argmax(model.predict(X_new), axis=-1)
    print(y_preds)
    response = {"sentiment": y_preds}
    return flask.jsonify(response)
