# Simple flask wrapper around a pretrained model

Trained on this dataset https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data

## Setup
cd into top level folder

docker build --tag sentiment-analysis -f Dockerfile .

docker run --env PORT=5000 sentiment-analysis


## Endpoint

endpoint is <url>/api/v1/get_sentiment?phrase=example phrase
  
returns one of 

   - NEGATIVE
   - SOMEWHAT_NEGATIVE
   - NEUTRAL
   - SOMEWHAT_POSITIVE
   - POSITIVE