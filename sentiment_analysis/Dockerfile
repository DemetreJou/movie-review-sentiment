FROM python:3.9.6


COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt
RUN pip install gunicorn

COPY ./ /app/sentiment_analysis

ENV PYTHONPATH "${PYTHONPATH}:/app/sentiment_analysis"

# PORT is set by heroku, this is only used in production deploy
WORKDIR /app/sentiment_analysis
CMD gunicorn --bind 0.0.0.0:$PORT wsgi