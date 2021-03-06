# irisdata-chainer

Predict iris species using Chainer.

# Env

- Docker version 18.03.1-ce, build 9ee9f40

# Usage

```
# Clone repository
$ git clone git@github.com:masayuki5160/irisdata-chainer.git
$ cd irisdata-chainer/

# Build Docker image
$ docker build -t masayuki5160/iris-chainer .

# Start Docker container
$ docker run -v $(pwd):/home/irisdata-chainer/ -it masayuki5160/iris-chainer /bin/bash 
$ cd /home/irisdata-chainer

# neural network
$ python iris_nn.py
# use model(iris_nn.npz)
$ python load_nn.py 

# logistic regression
$ python iris_logistic.py
# use model(iris_logistic.npz)
$ python iris_logistic.py 
```

# Appendix

* [Chainer v2による実践深層学習](https://www.ohmsha.co.jp/book/9784274221071/)
* [ディープラーニングがわかる数学入門](https://www.amazon.co.jp/%E3%83%87%E3%82%A3%E3%83%BC%E3%83%97%E3%83%A9%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0%E3%81%8C%E3%82%8F%E3%81%8B%E3%82%8B%E6%95%B0%E5%AD%A6%E5%85%A5%E9%96%80-%E6%B6%8C%E4%BA%95-%E8%89%AF%E5%B9%B8/dp/477418814X)
