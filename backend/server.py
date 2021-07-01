from flask import Flask

from routes import *
from sentiment_analysis.train_model import SentimentModel

model = SentimentModel(load_pretrained=True)

app = Flask(__name__)
app.register_blueprint(routes)


# run this for dev
if __name__ == '__main__':
    app.run(use_reloader=False, port=5000, threaded=True)
