from flask import Flask
from . import routes

@routes.route("/me/home")
def me_home():
    return "<h1> Welcome to my website </h1>"
