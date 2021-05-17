import time
from flask import Flask
from flask_cors import cross_origin
from flask import request
from . import routes
from ..server import model
from keras.preprocessing.text import Tokenizer


@routes.route('/api/v1/get_sentiment', methods=['GET'])
@cross_origin()
def get_sentiment():
    # TODO
    # process input
    # figure out a way to pass through tokenizer used in training (pickle?)
    # import model from server.py
    # pass through model
    # maybe convert from label to value

    # seq = tokenizer.texts_to_sequences(input)
    # X_new = sequence.pad_sequences(seq, maxlen=len_max)
    # y_preds = model.predict_classes(X_new

    return 0