FROM python:3.9.6


COPY backend/requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY ./backend /app/backend
COPY ./sentiment_analysis/  /app/sentiment_analysis

ENV PYTHONPATH "${PYTHONPATH}:/app/backend:/app/sentiment_analysis"

ENTRYPOINT [ "python" ]

CMD [ "backend/wsgi.py" ]