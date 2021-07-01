from flask import Blueprint

routes = Blueprint('routes', __name__)

from .time import *
from .sentiment_analysis import *
from .homepage import *
