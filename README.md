# A simple react + flask project

## Setup

Install docker, python, react

backend/requirements.txt is for both backend flask server as well as for /sentiment_analysis 
- can probably soft link backend/requirements.txt to sentiment_analysis/requirements.txt for ease


## TODO
Setup different environment to differentiate between dev and deploy environment

Convert backend dockerfile to a more lightweight container (build from python instead of ubuntu)

When training model validation accuracy per epoch isn't printing out properly

Include parts of speech tags when training, requirements changing model to functional model instead of sequential