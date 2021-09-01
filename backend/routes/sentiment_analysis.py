from flask import jsonify, request
import requests
from flask_cors import cross_origin
from . import routes


@routes.route('/api/v1/get_sentiment', methods=['GET'])
@cross_origin()
def get_sentiment():
    phrase = request.args.get("phrase")
    response = requests.get("http://localhost:5100/api/v1/get_sentiment?phrase=" + phrase)
    return response.json()
