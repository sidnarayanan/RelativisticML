#!/usr/bin/env python

import numpy as np
import theano
import theano.tensor as T

theano.config.int_division = 'floatX'

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
  def __getstate__(self):
      return {'W':self.W, 'b':self.b}
  def __setstate__(self,d):
      self.W = d['W']
      self.b = d['b']
  def NLL(self,y):
    '''
    NLL loss
    this assumes the classes are indexed 0...N
    '''
    return -T.mean(T.log(self.P)[T.arange(y.shape[0]),y])
  def aNLL(self,y):
    '''
    Asymmetric NLL loss
    gives more weight to classes with higher numbers
      (i.e. y=1 errors are penalized more than y=0)
    this assumes the classes are indexed 0...N
    '''
    return -T.mean((1*y+1)*T.log(self.P)[T.arange(y.shape[0]),y])
  def MSE(self,y):
    '''
    Mean square error loss
    only works for signal/background discrimination currently
    '''
    return T.mean((self.P[T.arange(y.shape[0]),1]-y)**2)
  def BoverS2(self,y):
      '''
      sqrt(B)/S loss, computed as B/S**2
      '''
      return T.sum(self.P[T.arange(y.shape[0]),1]*(1-y)) / (T.sum(self.P[T.arange(y.shape[0]),1]*y)**2)
  def BGReg(self,y):
      '''
      crappy regularization
      '''
      probs = self.P[T.arange(y.shape[0]),1]
      return T.max(probs*(1-y)) - T.min(probs*(1-y))
  def BGBinnedReg(self,y,varBinned):
    '''
    regularizes background efficiency in a binned variable var
    '''
    # baseHist = 0
    probs = self.P[T.arange(y.shape[0]),1]
    baseHist = T.bincount(varBinned,1-y)+0.01
    selectedHist = T.bincount(varBinned,(1-y)*probs)+0.01
    ratioHist = selectedHist/baseHist
    rVal = ratioHist - T.mean(ratioHist)
    # rVal = T.std(selectedHist/baseHist)
    return T.sum(rVal*rVal)
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
    elif lossType=='MSE':
      loss = self.MSE(trainY)
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
