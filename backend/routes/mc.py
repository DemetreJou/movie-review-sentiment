from flask import Flask
from . import routes

@routes.route("/mc/home")
def mc_home():
    return "<h1> Welcome to a soon to be side project </h1>"
