from flask import Blueprint
routes = Blueprint('routes', __name__)

from .mc import *
from .me import *