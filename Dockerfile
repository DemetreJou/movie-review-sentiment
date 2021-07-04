FROM python:3.9.6


COPY backend/requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt
RUN pip install gunicorn

COPY ./backend /app/backend
COPY ./sentiment_analysis/  /app/sentiment_analysis

ENV PYTHONPATH "${PYTHONPATH}:/app/backend:/app/sentiment_analysis"

# PORT is set by heroku, this is only used in production deploy
CMD gunicorn --bind 0.0.0.0:$PORT wsgi