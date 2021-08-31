from flask import Blueprint

routes = Blueprint('routes', __name__)

# pylint: disable=wrong-import-position
from .time import *
from .sentiment_analysis import *
from .homepage import *
