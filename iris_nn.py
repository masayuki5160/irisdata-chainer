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
# Xの奇数行をxtrainとする
xtrain = X[index[index % 2 != 0],:]
# Y2の奇数行をytrainとする
ytrain = Y2[index[index % 2 != 0],:]
# Xの偶数行をxtestとする
xtest = X[index[index % 2 == 0],:]
# Yの偶数行をyansとする
yans = Y[index[index % 2 == 0]]

train = tuple_dataset.TupleDataset(xtrain, ytrain)

# Define model
# モデルを定義するためChainクラスを継承する
class IrisChain(Chain):
    def __init__(self):
        super(IrisChain, self).__init__(
            # Define layes
            l1=L.Linear(4,6),
            l2=L.Linear(6,3),
        )
    
    # Cost function
    # 誤差の総和の計算(損失関数、コスト関数、誤差関数などいろいろよばれる)
    def __call__(self,x,y):
        # 訓練データで順伝播した計算結果と教師データから誤差を計算
        return F.mean_squared_error(self.fwd(x), y)
    
    # 順伝播
    def fwd(self,x):
        # 活性化関数としてシグモイド関数を利用し出力ベクトルを得る
        h1 = F.sigmoid(self.l1(x))
        # TODO 活性化関数はいらない？
        h2 = self.l2(h1)
        return h2
     
# Initialize model
model = IrisChain()
optimizer = optimizers.SGD()
optimizer.setup(model)

# learn by trainer 
train_iter = iterators.SerialIterator(train, 25)
updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (5000, 'epoch'))
trainer.extend(extensions.ProgressBar())
trainer.run()

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

# Save model
model.to_cpu()
serializers.save_npz("iris_nn.npz", model)
