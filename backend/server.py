from flask import Flask

# pylint: disable=unused-wildcard-import, unused-import, wildcard-import
from routes import *

app = Flask(__name__)
app.register_blueprint(routes)


# run this for dev
if __name__ == '__main__':
    app.run(use_reloader=False, port=5000, threaded=True)
