FROM ubuntu:20.10

RUN apt-get update -y
RUN apt-get install python3.8 python3-pip -y

COPY backend/requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install -r requirements.txt

COPY ./backend /app
COPY ./sentiment_analysis/trained_model /app/trained_model

ENTRYPOINT [ "python3" ]

CMD [ "wsgi.py" ]