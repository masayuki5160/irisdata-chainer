#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.datasets import tuple_dataset
from chainer import training
from chainer.training import extensions

# Set data
X = np.loadtxt('iris-x.txt').astype(np.float32)
Y = np.loadtxt('iris-y.txt').astype(np.int32)
N = Y.size
Y2 = np.zeros(3 * N).reshape(N,3).astype(np.float32)
for i in range(N):
    Y2[i,Y[i]] = 1.0

index = np.arange(N)
xtrain = X[index[index % 2 != 0],:]
ytrain = Y2[index[index % 2 != 0],:]
xtest = X[index[index % 2 == 0],:]
yans = Y[index[index % 2 == 0]]

train = tuple_dataset.TupleDataset(xtrain, ytrain)

# Define model
# モデルを定義するためChainクラスを継承する
class IrisChain(Chain):
    def __init__(self):
        super(IrisChain, self).__init__(
            # Define layes
            l1=L.Linear(4,3),
        )
    
    # Cost function
    # 誤差の総和の計算(損失関数、コスト関数、誤差関数などいろいろよばれる)
    def __call__(self,x,y):
        # 訓練データで順伝播した計算結果と教師データから誤差を計算
        return F.mean_squared_error(self.fwd(x), y)
    
    # 順伝播
    def fwd(self,x):
        return F.softmax(self.l1(x))
     
# Initialize and load model
model = IrisChain()
serializers.load_npz("iris_logistic.npz", model)

# Test
xt = Variable(xtest)
yy = model.fwd(xt)

ans = yy.data
nrow, ncol = ans.shape
ok = 0
for i in range(nrow):
    cls = np.argmax(ans[i,:])
    print ans[i,:], cls            
    if cls == yans[i]:
        ok += 1
        
print ok, "/", nrow, " = ", (ok * 1.0)/nrow

