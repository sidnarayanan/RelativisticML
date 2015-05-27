#!/usr/bin/env python

import numpy as np
import theano
import theano.tensor as T


class Logistic(object):
  '''Logistic regression, maximizes $p_j \propto  w_{ij}\cdot x_i+b_j$'''
  def __init__(self,nIn,nOut,x):
    self.W = theano.shared(value=np.zeros((nIn,nOut),dtype=theano.config.floatX), name='W',borrow=True)    
    self.b = theano.shared(value=np.zeros((nOut,),dtype=theano.config.floatX), name='b',borrow=True)
    self.input = x
    self.P = T.nnet.softmax(T.dot(self.input,self.W) + self.b)
    self.yHat = T.argmax(self.P,axis=1)
    testX = T.matrix('testX')
    testY = T.ivector('testY')
    self.evaluate = theano.function(
        inputs = [testX],
        outputs = self.yHat,
        givens = {
          self.input : testX
        }
      )
    self.errors = theano.function(
        inputs = [testX,testY],
        outputs = T.mean(T.neq(self.yHat,testY)),
        givens = {
          self.input : testX
        },
        allow_input_downcast=True
      )
    self.probabilities = theano.function(
        inputs = [testX],
        outputs = self.P,
        givens = {
          self.input : testX
        }
      )
  def NLL(self,y):
    '''
    NLL loss
    this assumes the classes are indexed 0...N
    '''
    return -T.mean(T.log(self.P)[T.arange(y.shape[0]),y])
  def aNLL(self,y):
    '''
    Asymmetric NLL loss
    gives more weight to classes with higher numbers (i.e. y=1 errors are penalized more than y=0)
    this assumes the classes are indexed 0...N
    '''
    return -T.mean((2*y+1)*T.log(self.P)[T.arange(y.shape[0]),y])
  def getTrainer(self,lossType="NLL"):
    '''
    return a function to do MBSGD on (trainX,trainY) 
    '''
    
    trainY = T.ivector('y')
    alpha = T.dscalar('a')
    lowIdx = T.iscalar()
    highIdx = T.iscalar()
    trainX = T.matrix()
    if lossType=="aNLL":
      loss = self.aNLL(trainY)
    else:
      loss = self.NLL(trainY)
    dW = T.grad(cost = loss, wrt = self.W)
    db = T.grad(cost = loss, wrt = self.b)
    updates = [(self.W,self.W - alpha * dW), (self.b,self.b - alpha * db)]
    trainer = theano.function(
        inputs = [trainX,trainY,alpha],
        outputs = loss,
        updates=updates,
        givens = {
          self.input : trainX,
        },
        allow_input_downcast=True
      )
    return trainer