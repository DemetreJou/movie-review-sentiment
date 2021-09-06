import os

from flask import Flask

# pylint: disable=unused-wildcard-import, unused-import, wildcard-import
from routes import *
from dotenv import load_dotenv, dotenv_values
from os.path import join, dirname


if os.getenv("IN_DOCKER", False):
    dotenv_path = join(dirname(__file__), "configs", ".docker-env")
else:
    dotenv_path = join(dirname(__file__), "configs", ".local-env")

SETTINGS = dotenv_values(dotenv_path=dotenv_path)

app = Flask(__name__)
app.register_blueprint(routes)


# run this for dev
if __name__ == '__main__':
    # TODO: fix this reloader, reloads on every change in /sentiment_analysis instead of just those in /backend
    app.run(use_reloader=False, port=5000, threaded=True)
