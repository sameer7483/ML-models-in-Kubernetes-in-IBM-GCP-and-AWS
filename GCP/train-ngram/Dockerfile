# Start with a Linux micro-container to keep the image tiny
# FROM alpine:3.7
FROM ubuntu:18.04

# Document who is responsible for this image
MAINTAINER Ritika "rm6383@nyu.edu"

# Install just the Python runtime (no dev)
RUN apt-get -s autoclean
RUN apt-get update
# Install just the Python runtime (no dev)
RUN  apt-get install -y \
    python3.6 python3-dev gcc \
    ca-certificates curl

RUN curl -fsSL https://bootstrap.pypa.io/pip/3.6/get-pip.py | python3.6


# Set up a working folder and install the pre-reqs
WORKDIR /app
ADD requirements.txt /app
RUN pip3 install --upgrade pip setuptools 
RUN pip3 --no-cache-dir install -r requirements.txt


# Add the code as the last Docker layer because it changes the most
ADD main.py  /app/main.py

# Run the service
CMD [ "python3", "main.py"]
