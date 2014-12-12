#!/usr/bin/env python


import numpy as np
import sys

exp = np.exp

#default is logistic sigmoid, tanh etc can be used if desired
# nevermind use tanh, logistic sigmoid sucks
def sigmoid(w):
  r = 1./(1.+exp(-w))
  return r

def dsigmoid(w):
  return (1.-sigmoid(w))*sigmoid(w)

tanh=np.tanh
def mtanh(w):
  # modified tanh as per LeCunn S4.4
  return 1.7159*tanh(2.*w/3)
  # return tanh(w)

def dmtanh(w):
  # derivative
  return (2/3)*1.7159*(1-pow(tanh(2*w/3),2.))
  # return 1-pow(tanh(w),2.)

def dtanh(w):
  return 1-pow(tanh(w),2.)

class Neuron(object):
  def __init__(self,neuronType, sw=0):
    self.sumw=sw
    self.error=0
    if neuronType=="output":
      self.transfer=sigmoid
      self.dtransfer=dsigmoid
    else:
      self.transfer=tanh
      self.dtransfer=dtanh
    self.delta=0
    self.neuronType=neuronType
  def getGradient(self):
    return self.dtransfer(self.sumw)
  def getDelta(self):
    return self.delta
  def setDelta(self,d):
    if self.neuronType=="output":
      # self.delta=d
      self.delta=self.getGradient()*d
    else:
      self.delta=self.getGradient()*d
  def evaluate(self):
    # if self.neuronType=="input" or self.neuronType=="output":
    if self.neuronType=="input":
      # only hidden nodes get sigmoided 
      return self.sumw
    else:
      return self.transfer(self.sumw)

class Layer(object):
  def __init__(self,t,n=0):
    self.nNodes=n
    self.layerType=t
    self.nodes=[]
    self.addNeurons(self.nNodes)
  def addNeurons(self,n):
    if n==0:
      return
    else:
      # self.nNodes+=1
      if self.layerType=="input":
        neuron = Neuron(self.layerType)
        # neuron = Neuron( lambda x : x, lambda x : 1)
      else:
        neuron = Neuron(self.layerType)
      self.nodes.append(neuron)
      self.addNeurons(n-1)
  def getErrors(self):
    return np.array(map( lambda x: x.error , self.nodes ))
  def getSumW(self):
    return np.array(map( lambda x: x.sumw , self.nodes ))
  def getDeltas(self):
    return np.array(map( lambda x: x.getDelta(), self.nodes ))
  def setDeltas(self,wd):
    for n,w in zip(self.nodes,wd):
      n.setDelta(w)
  def setSumW(self,weights):
    # if self.layerType=="output":
    #   print "setting output sumw:",weights
    for n,w in zip(self.nodes,weights):
      n.sumw=w
  def setSumErrors(self,errors):
    for n,e in zip(self.nodes,errors):
      n.error=e
  def evaluate(self):
    # if self.layerType=="hidden":
    #   print "hidden eval",np.array( map( lambda x: x.evaluate() , self.nodes )) 
    return np.array( map( lambda x: x.evaluate() , self.nodes ))


class Network(object):
  # still need to implement learning momentum
  def __init__(self,nl,dims,a=1,p=0):
    self.alpha=a # learning rate
    self.p=p # momentum
    self.nLayers=nl
    self.layers=[]
    self.weights=[]
    self.weightDelta=[] # internal representation of weight changes - needs implementation
    self.dims=dims
    for n in range(self.nLayers):
      if n==0:
        l = Layer("input",dims[n])
      elif n==self.nLayers-1:
        l = Layer("output",dims[n])
        self.weights.append(np.random.random([dims[n-1],dims[n]]))
        self.weights[n-1] = self.weights[n-1]/dims[n-1]
      else:
        l = Layer("hidden",dims[n])
        self.weights.append(np.random.random([dims[n-1],dims[n]]))
        self.weights[n-1] = self.weights[n-1]/dims[n-1]
      self.layers.append(l)
      # print self.weights
    self.inputLayer=self.layers[0]
    self.outputLayer=self.layers[self.nLayers-1]
    self.hiddenLayers = self.layers[1:self.nLayers-1]
  def __str__(self):
    s="Network Class Summary:\n\talpha=%f\n\tp=%f\n\tnLayers=%i\n\tdims=%s\n"%(self.alpha,self.p,self.nLayers,str(self.dims))
    return s
  def evaluate(self,x):
    self.inputLayer.setSumW(x)
    for n in range(self.nLayers-1):
      # print n,self.layers[n].evaluate()
      # print self.weights[n]
      prop=np.dot(self.layers[n].evaluate(),self.weights[n])
      self.layers[n+1].setSumW(prop)
    return self.outputLayer.evaluate()
  def backProp(self,x,y):
    for n in range(self.nLayers-1,-1,-1):
      if n==self.nLayers-1:
        yhat = self.evaluate(x)
        # print "outputs",yhat,y
        self.outputLayer.setDeltas(yhat-y)
        # print "delta[%i]"%(n),self.outputLayer.getDeltas()
      else:
        z = self.layers[n].evaluate()
        delta = self.layers[n+1].getDeltas()
        # print "delta ",delta
        self.layers[n].setDeltas(np.dot(self.weights[n],delta))
        # sys.stderr.write("z %i %s\n"%(n,str(z)))
        # sys.stderr.write("delta %i %s\n"%(n+1,str(delta)))
        # print z,delta
        # print "delta[%i]"%(n+1),delta
        # print "z[%i]"%(n),z
        self.weights[n] -= np.outer(z,delta)*self.alpha
        # self.weights[n] = self.weights[n]/self.weights[n].sum(axis=0) # maintain normalization
  def evaluateOnSet(self,X,Y,nSkip=0,nDo=-1):
    # for when you want to batch evaluate
    # returns yhat and sum_i (yhat_i-y_i)^2
    i=0
    Yhat=[]
    err=0
    for x,y in zip(list(X)[nSkip:],Y[nSkip:]):
      xhat = np.concatenate((x,[1])) # add constant offset
      yhat = self.evaluate(xhat)
      Yhat.append(yhat)
      err += pow((y-yhat),2.)
    return Yhat,err



