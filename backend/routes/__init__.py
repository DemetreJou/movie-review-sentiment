from flask import Blueprint
routes = Blueprint('routes', __name__)

from .me import *
from .time import *
from .sentiment_analysis import *