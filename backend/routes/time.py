import time
from flask_cors import cross_origin
from . import routes


# endpoint to help with testing backend is running without relying on the model being loaded correctly
@routes.route('/api/v1/time')
@cross_origin()
def get_current_time():
    return {'time': time.time()}
