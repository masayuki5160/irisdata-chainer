FROM chainer/chainer:latest

COPY . /irisdata-chainer

RUN pip install chainer numpy 
