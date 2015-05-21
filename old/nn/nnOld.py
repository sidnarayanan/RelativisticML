#!/usr/bin/env python

import numpy as np
import sys
import cPickle as pickle
import pathos.multiprocessing as mp
from multiprocessing import cpu_count
exp = np.exp

def sigmoid(w):
  r = 1./(1.+exp(-w))
  return r

def dsigmoid(w):
  return (1.-sigmoid(w))*sigmoid(w)

tanh=np.tanh
def mtanh(w):
  return 1.7159*tanh(2.*w/3)

def dmtanh(w):
  # derivative
  return (2/3)*1.7159*(1-pow(tanh(2*w/3),2.))

def dtanh(w):
  return 1-pow(tanh(w),2.)

class Neuron(object):
  def __init__(self,neuronType, sw=0):
    self.sumw=sw
    self.error=0
    if neuronType=="output":
      self.transfer= sigmoid
      self.dtransfer= dsigmoid
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
      self.delta=self.getGradient()*d
    else:
      self.delta=self.getGradient()*d
  def evaluate(self):
    if self.neuronType=="input":
      # only hidden and output nodes get sigmoided 
      return self.sumw
    else:
      # print self.sumw,self.transfer(self.sumw)
      # sys.exit(-1)
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
    for n,w in zip(self.nodes,weights):
      n.sumw=w
  def setSumErrors(self,errors):
    for n,e in zip(self.nodes,errors):
      n.error=e
  def evaluate(self):
    return np.array( map( lambda x: x.evaluate() , self.nodes ))


class Network(object):
  # still need to implement learning momentum
  def __init__(self,dims,data=(None,None),a=1,p=0):
    self.alpha=0.5 # learning rate
    self.p=p # momentum
    self.nLayers=len(dims)
    self.layers=[]
    self.weights=[]
    self.oldWeights=[]
    self.dims=dims
    self.dataX = data[0]
    self.dataY = data[1]
    self.trainingIndices = None
    self.testingIndices = None
    self.validationIndices = None
    self.debug = False
    self.summaryFile = None
    self.dataHasBeenShuffled = False
    for n in range(self.nLayers):
      if n==0:
        l = Layer("input",dims[n])
      elif n==self.nLayers-1:
        l = Layer("output",dims[n])
        self.weights.append( np.random.random((dims[n-1],dims[n])) )
        self.weights[n-1] = self.weights[n-1]/dims[n-1]
      else:
        l = Layer("hidden",dims[n])
        self.weights.append( np.random.random((dims[n-1],dims[n])) )
        self.weights[n-1] = self.weights[n-1]/dims[n-1]
      self.layers.append(l)
    self.inputLayer=self.layers[0]
    self.outputLayer=self.layers[self.nLayers-1]
    self.hiddenLayers = self.layers[1:self.nLayers-1]

  def createSummaryFile(self,fileName):
    self.summaryFile = open(fileName,"w")
  def destroySummaryFile(self):
    self.summaryFile.close()

  def __str__(self):
    s="Network Class Summary:\n\talpha=%f\n\tp=%f\n\tnLayers=%i\n\tdims=%s\n"%(self.alpha,self.p,self.nLayers,str(self.dims))
    return s

  def printSummary(self):
    self.summaryFile.write(str(self))

  def evaluate(self,x):
    self.inputLayer.setSumW(x)
    for n in range(self.nLayers-1):
      prop=np.dot(self.layers[n].evaluate(),self.weights[n])
      self.layers[n+1].setSumW(prop)
    return self.outputLayer.evaluate()

  def backProp(self,x,y):
    for n in range(self.nLayers-1,-1,-1):
      if n==self.nLayers-1:
        yhat = self.evaluate(x)
        self.outputLayer.setDeltas(yhat-y)
      else:
        z = self.layers[n].evaluate()
        delta = self.layers[n+1].getDeltas()
        self.layers[n].setDeltas(np.dot(self.weights[n],delta))
        self.weights[n] -= np.outer(z,delta)*self.alpha
        self.weights[n] = self.weights[n]/self.weights[n].sum(axis=0) # maintain normalization

  def normalizeWeights(self):
    for n in range(self.nLayers-2,-1,-1):
      self.weights[n] = self.weights[n]/self.weights[n].sum(axis=0)

  # def hiddenBackProp(self,index):
  #   error = self.evaluateError(index)
  #   # do backprop but do not apply weight differences
  #   weightDelta = []
  #   for n in range(self.nLayers-1):
  #     weightDelta.append(None)
  #   for n in range(self.nLayers-1,-1,-1):
  #     if n==self.nLayers-1:
  #       self.outputLayer.setDeltas(error)
  #     else:
  #       z = self.layers[n].evaluate()
  #       delta = self.layers[n+1].getDeltas()
  #       self.layers[n].setDeltas(np.dot(self.weights[n],delta))
  #       weightDelta[n] = np.outer(z,delta)*self.alpha
  #   return weightDelta

  # def evaluateError(self,i):
  #   xhat = np.concatenate((self.dataX[i],[1]))
  #   return self.evaluate(xhat) - self.dataY[i]

  def evaluatePerformance(self,indices):
    t=0.
    for i in indices:
      xhat = np.concatenate((self.dataX[i],[1]))
      t+=  pow(self.evaluate(xhat) -  self.dataY[i] , 2)
    t = t/len(indices)
    return np.sqrt(t)

  def testNetwork(self):
    return self.evaluatePerformance(self.testingIndices)
  def validateNetwork(self):
    return self.evaluatePerformance(self.validationIndices)

  # def trainSGD(self):
  #   self.alpha = 0.1
  #   if not self.dataHasBeenShuffled:
  #     # will refuse to train on unshuffled data, duh
  #     self.randomizeIndices()
  #   testError = self.testNetwork()
  #   lastError = testError
  #   nEpoch=0
  #   i=0
  #   for idx in self.trainingIndices:
  #     if nEpoch>25:
  #       break
  #     x=self.dataX[idx]
  #     y=self.dataY[idx]
  #     if not i%10000:
  #       # stderr.write("training %i\n"%(i))
  #       sys.stdout.write("training %i\n"%(i))
  #     i+=1
  #     yhat = self.evaluate(x)
  #     self.backProp(x,y)
  #     if not i%10000:
  #     # if True:
  #       nEpoch+=1
  #       try:
  #         print yhat,y,abs(float(y)-yhat)
  #         # print "error ",abs(float(y)-yhat)
  #       except RuntimeWarning:
  #         # probably means something went out of bound in the network, no point continuing
  #         # this does not work fix it later
  #         print y,yhat
  #         exit(-1)
  #       print self.testNetwork()

  # def train(self,batchSize=1):
  #   if not self.dataHasBeenShuffled:
  #     # will refuse to train on unshuffled data, duh
  #     self.randomizeIndices()
  #   self.alpha=1
  #   testError = self.testNetwork()
  #   lastError = testError
  #   nEpoch=0
  #   if self.debug:
  #     self.summaryFile.write("Epoch: %i\t Error: %f\n"%(nEpoch,testError))
  #   j=0
  #   nTrain = len(self.trainingIndices)
  #   print nTrain/batchSize
  #   pool = mp.ProcessingPool(cpu_count()-1)
  #   nIncrease=0 
  #   nDecrease=0
  #   isOscillating=False
  #   errors=[]
  #   errors.append(testError)
  #   while True:
  #     batchWeightDeltas = pool.map( self.hiddenBackProp , self.trainingIndices[j*batchSize:(j+1)*batchSize])
  #     for i in range(batchSize):
  #       for k in range(self.nLayers-1):
  #         self.weights[k] -= batchWeightDeltas[i][k]/float(batchSize)
  #     self.normalizeWeights()
  #     lastError = testError
  #     testError = self.testNetwork()
  #     errors.append(testError)
  #     if testError > lastError:
  #       nIncrease+=1
  #       nDecrease=0
  #     else:
  #       nDecrease+=1
  #       nIncrease=0
  #     if self.debug:
  #         self.summaryFile.write("Epoch: %i\tBatch: %i\t Error: %f\n"%(nEpoch,j,testError))
  #         sys.stdout.write("Epoch: %i\tBatch: %i\t Error: %f\n"%(nEpoch,j,testError))
  #     j = (j+1) % (nTrain/batchSize)
  #     if j==0:
  #       nEpoch+=1
  #     if nIncrease > 1 or nDecrease>1:
  #       isOscillating = False
  #     if nIncrease>4:
  #       self.alpha = self.alpha/2.
  #       self.summaryFile.write("Lowering alpha to %.4f\n"%(self.alpha))
  #       sys.stdout.write("Lowering alpha to %f\n"%(self.alpha))
  #       if self.alpha < 0.0000001:
  #         break
  #     elif len(errors)>4 and not isOscillating:
  #       if (errors[-4] > errors[-3] and errors[-2]>errors[-3] and errors[-2]>errors[-1]) or (errors[-4]<errors[-3] and errors[-2]<errors[-3] and errors[-2]<errors[-1]):
  #         isOscillating = True
  #         # oscillatory behavior
  #         self.alpha = self.alpha/2.
  #         self.summaryFile.write("Lowering alpha to %.4f\n"%(self.alpha))
  #         sys.stdout.write("Lowering alpha to %f\n"%(self.alpha))
  #         if self.alpha < 0.0000001:
  #           break
          
  def randomizeIndices(self,nTrain=-1, nTest=-1, nValidate=-1):
    nData = self.dataY.shape[0]
    if nTrain==-1:
      nTrain = nData/3
      nTest = nTrain
      nValidate = nTrain
    elif nTest==-1:
      nTest = (nData-nTrain)/2
      nValidate = nTest
    elif nValidate==-1:
      nValidate = nData-nTrain-nTest
    indices = range(nData)
    np.random.shuffle(indices)
    self.trainingIndices = indices[:nTrain]
    self.testingIndices = indices[nTrain:nTrain+nTest]
    self.validationIndices = indices[nTrain+nTest:nTrain+nTest+nValidate]
    self.dataHasBeenShuffled = True

  def dumpToPickle(self,pickleJarName):
    pickleJar = open(pickleJarName,'wb')
    pickleDict = {"dataX":self.dataX,
                  "dataY":self.dataY,
                  "dims":self.dims,
                  "weights":self.weights,
                  "alpha":self.alpha,
                  "nLayers":self.nLayers,
                  "dataHasBeenShuffled":self.dataHasBeenShuffled,
                  "trainingIndices":self.trainingIndices,
                  "testingIndices":self.testingIndices,
                  "validationIndices":self.validationIndices}
    pickle.dump(pickleDict,pickleJar,2)
    pickleJar.close()

  def loadFromPickle(self,pickleJarName):
    pickleJar = open(pickleJarName,'rb')
    pickleDict = pickle.load(pickleJar)
    self.dataX = pickleDict["dataX"]
    self.dataY = pickleDict["dataY"]
    self.dims = pickleDict["dims"]
    self.weights = pickleDict["weights"]
    self.alpha = pickleDict["alpha"]
    self.nLayers = pickleDict["nLayers"]
    self.dataHasBeenShuffled = pickleDict["dataHasBeenShuffled"]
    self.trainingIndices = pickleDict["trainingIndices"]
    self.testingIndices = pickleDict["testingIndices"]
    self.validationIndices = pickleDict["validationIndices"]