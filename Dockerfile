FROM chainer/chainer:latest

VOLUME /home/irisdata-chainer

RUN apt-get update && \
    apt-get -y install vim && \
    pip install chainer numpy 
