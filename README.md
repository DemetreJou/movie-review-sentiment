# A simple react + flask project

## Dataset
Download from here 
https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data

## Setup
This is run dockerized for easy production deployment(also because I wanted docker practice)

I'm self hosting everything even if heroku is easier to startup on (also because I wanted the practice)

Everything is dockerized but can be run standalone locally

Going to run frontend and backend in different containers, use traefik as reverse proxy when deployed, no need for database yet

## TODO
Setup different environment to differentiate between dev and deploy environment

Find solution for passing tokenizer from training to web app (probably pickle)

It actually seems like training takes longer when using CUDA, should investigate

Try finding an efficient way of including parts-of-speech tags as a way to hopefully improve accuracy

Convert backend dockerfile to a more lightweight container (build from python instead of ubuntu)s
