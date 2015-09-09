#!/usr/bin/env python

import theano
import theano.tensor as T
import numpy as np
from Logistic import *
import sys,os


x = T.matrix('x')

classifier = Logistic(2,2,x)

trainX = np.zeros([1000000,2])
trainY = np.zeros([1000000,1]).astype(int)
for i in range(1000000):
  trainX[i,i%2]=1
  trainY[i]=i%2
trainer = classifier.getTrainer()
# print trainer(lowIdx =0, highIdx=50,a=.1)
for i in range(10000):
  r =  trainer(trainX[i*50:(i+1)*50],trainY[i*50:(i+1)*50][:,0],.1)
  print r
print classifier.errors(trainX[500000:],trainY[500000:][:,0])
print classifier.evaluate(trainX[999990:]),trainY[999990:].T