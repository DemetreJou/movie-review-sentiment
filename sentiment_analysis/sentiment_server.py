from flask import Flask, jsonify, request

app = Flask(__name__)

from pipeline.train_model import SentimentModel

model = SentimentModel(load_pretrained=True)


@app.route('/api/v1/get_sentiment', methods=['GET'])
def get_sentiment():
    phrase = request.args.get("phrase")
    sentiment = model.get_sentiment(phrase).name
    return jsonify(sentiment)


@app.route('/api/v1/', methods=['GET'])
def get_homepage():
    return "Hello World!"


# run this for dev
if __name__ == '__main__':
    app.run(use_reloader=False, port=5100, threaded=True)
