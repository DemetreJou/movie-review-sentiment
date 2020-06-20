import os
from flask import Flask, send_from_directory, render_template
from routes import *

app = Flask(__name__, static_folder='../frontend/build')
app.register_blueprint(routes)

# Serve React Apps
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route("/taylor")
def baby():
    return "<h1> You're baby !! <3 </h1>"


if __name__ == '__main__':
    app.run(use_reloader=False, port=5000, threaded=True)
