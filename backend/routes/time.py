import time
from flask import Flask
from flask_cors import cross_origin
from . import routes

@routes.route('/api/v1/time')
@cross_origin()
def get_current_time():
    return {'time': time.time()}