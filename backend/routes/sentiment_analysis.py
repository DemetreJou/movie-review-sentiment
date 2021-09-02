from flask import jsonify, request
import requests
from flask_cors import cross_origin
from . import routes
from requests_toolbelt import sessions
from utils.network_request import request_client_generator


request_instance = request_client_generator(base_url="http://localhost:5100/api/v1/")

@routes.route('/api/v1/get_sentiment', methods=['GET'])
@cross_origin()
def get_sentiment():
    phrase = request.args.get("phrase")
    response = request_instance.get("http://localhost:5100/api/v1/get_sentiment?phrase=" + phrase, timeout=10)
    return response.json()
