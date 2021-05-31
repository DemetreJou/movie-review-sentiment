import time
from flask_cors import cross_origin
from . import routes


@routes.route('/api/v1/get_sentiment', methods=['GET'])
@cross_origin()
def get_sentiment():
    # TODO: implement
    phrase = request.args.get("phrase")
    return flask.jsonify("Neutral")
