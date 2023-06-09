# Start with a Linux micro-container to keep the image tiny
# FROM alpine:3.7
FROM --platform=linux/amd64 python:3

# Document who is responsible for this image
MAINTAINER Sameer Ahmed "sa6142@nyu.edu"


WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ADD sentiment.py /app/sentiment.py


ADD train.py /app/train.py
ADD sentiment2.py /app/sentiment2.py

RUN mkdir ./templates
RUN mkdir ./data

ADD ./templates/index.html /app/templates/index.html

EXPOSE 5000
