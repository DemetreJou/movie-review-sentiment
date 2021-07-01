from flask import request, jsonify
from flask_cors import cross_origin

from backend.server import model
from . import routes


@routes.route('/api/v1/get_sentiment', methods=['GET'])
@cross_origin()
def get_sentiment():
    phrase = request.args.get("phrase")
    sentiment = model.get_sentiment(phrase).name
    return jsonify(sentiment)
