#!/usr/bin/env python

import theano
import theano.tensor as T
import numpy as np
from MLP import *
import sys,os
from sys import exit,stderr,stdout
import cPickle as pickle
import time
import re

interval = 86400*7*12 # look 3 months back
end = time.time()
start = end - interval
period = 86400*7

pklJar = open(sys.argv[1],'rb')
pklDict = pickle.load(pklJar)
try:
  dataX = pklDict['dataX']
  dataY = pklDict['dataY']
  uncertaintyX = pklDict['uncertaintyX']
except KeyError:
  datasetSet = pklDict['datasetSet']
  nDatasets = len(datasetSet)
  dataX = []
  uncertaintyX = []
  dataY = []
  for datasetName,datasetObject in datasetSet.iteritems():
    if not re.match(datasetPattern,datasetName):
      continue
    dataX.append(np.zeros(int(end-start)/period-1))
    dataY.append(0)
    for site,accesses in datasetObject.nAccesses.iteritems():
      for utime,n in accesses.iteritems():
        if utime < start or utime > end - period:
          # we don't want too recent data...might be sparse
          continue
        if utime > end - 2*period:
          # this is the point we'll try to predict 
          dataY[-1] += n
        else:
          dataX[-1][int((utime-start)/period)] +=  n
    uncertaintyX.append(np.mean(dataX[-1]))
    dataX[-1] = dataX[-1] - uncertaintyX[-1]
    dataY[-1] = dataY[-1] - uncertaintyX[-1]
    uncertaintyX[-1] = np.sqrt(uncertaintyX[-1])
    if dataY[-1] > uncertaintyX[-1]:
      dataY[-1] = 2
    elif dataY[-1] < uncertaintyX[-1]:
      dataY[-1] = 0
    else:
      dataY[-1] = 1
  dataY = np.array(dataY).astype(int)
  dataX = np.array(dataX)
  uncertaintyX = np.array(uncertaintyX)

x = T.matrix('x')
rng = np.random.RandomState()
classifier = MLP(x,rng,dataX.shape[1],40,1)
trainer = classifier.getTrainer(1,1)
nData = dataX.shape[0]
for i in range(nData/3*2/50):
  r =  trainer(trainX[i*50:(i+1)*50],trainY[i*50:(i+1)*50][:,0],.1)
  print r
print classifier.errors(trainX[nData/3*2:],trainY[nData/3*2:][:,0])
# print classifier.evaluate(trainX[999990:]),trainY[999990:].T