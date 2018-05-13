FROM chainer/chainer:latest

COPY . /irisdata-chainer

RUN apt-get update && \
    apt-get -y install vim && \
    pip install chainer numpy 
