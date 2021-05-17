from flask import Flask
from routes import *
import keras
import os

# this path should probably come through as an environment variable for difference when in docker container vs when local dev
model = keras.models.load_model(os.path.join("..", "sentiment_analysis", "trained_model"))

app = Flask(__name__)
app.register_blueprint(routes)


if __name__ == '__main__':
    app.run(use_reloader=False, port=5000, threaded=True)
