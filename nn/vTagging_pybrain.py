#!/usr/bin/env python

from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
import numpy as np
import ROOT
from sys import exit,stderr,stdout
from math import isnan

importWeights = False
printWeights =  True

listOfRawVars = ["fjet1QGtagSub1","fjet1QGtagSub2","fjet1QGtag","fjet1PullAngle","fjet1Pull","fjet1MassTrimmed","fjet1MassPruned","fjet1MassSDbm1","fjet1MassSDb2","fjet1MassSDb0","fjet1QJetVol","fjet1C2b2","fjet1C2b1","fjet1C2b0p5","fjet1C2b0p2","fjet1C2b0","fjet1Tau2","fjet1Tau1"]   
# listOfRawVars = ["fjet1QGtagSub1","fjet1QGtagSub2","fjet1Tau2","fjet1Tau1"]   
nRawVars = len(listOfRawVars)
sigFile = ROOT.TFile("../signal_word.root")
bgFile = ROOT.TFile("../background_word.root")
sigTree = sigFile.Get("DMSTree")
bgTree = bgFile.Get("DMSTree")
sigLeaves = sigTree.GetListOfLeaves()
bgLeaves = bgTree.GetListOfLeaves()
sigDict={}
bgDict={}
for i in range(sigLeaves.GetEntries()) :
  leaf = sigLeaves.At(i)
  sigDict[leaf.GetName()] = leaf
for i in range(bgLeaves.GetEntries()) :
  leaf = bgLeaves.At(i)
  bgDict[leaf.GetName()] = leaf

# nEvents=129912
# nEvents=99999999
# nEvents = min(nEvents,sigTree.GetEntries(),bgTree.GetEntries())
# nEvents = 2*(sigTree.GetEntries()+bgTree.GetEntries())/3
nEvents = 20000
# use nEvents/2 for training, nEvents/2 for validation (from each sample)
print "Training on %i events"%(nEvents)
trainX = np.empty([nEvents,nRawVars+2]) # two extra computed
trainY= np.zeros([nEvents])
sigcount=0
bgcount=0
for n in range(nEvents):
  #if not n%2: 
  if not n%2: 
    goodEvent=False
    while not goodEvent:
      sigTree.GetEntry(sigcount)
      m=0
      goodEvent=True
      for name in listOfRawVars:
        trainX[n,m]=sigDict[name].GetValue()
        if isnan(trainX[n,m]):
          print "WARNING, event ",sigcount,name," is nan in signal, skipping!"
          goodEvent=False
          break
        m+=1
      sigcount+=1
    trainY[n]=1
  else:
    goodEvent=False
    while not goodEvent:
      bgTree.GetEntry(bgcount)
      m=0
      goodEvent=True
      for name in listOfRawVars:
        trainX[n,m]=bgDict[name].GetValue()
        if isnan(trainX[n,m]):
          print "WARNING, event ",bgcount,name," is nan in bg, skipping!"
          goodEvent=False
          break
        m+=1
      bgcount+=1
  trainX[n,nRawVars] = 2*trainX[n,1] + trainX[n,0] # 2*fjet1QGtagSub2+fjet1QGtagSub1
  trainX[n,nRawVars+1] = float(trainX[n,nRawVars-2])/trainX[n,nRawVars-1] # tau2/tau1

trainX = (trainX - trainX.mean(0))/trainX.std(0) # center and normalize
print trainX[0]
print trainY[0]

ds = SupervisedDataSet(nRawVars+3,1)
for x,y in zip(list(trainX)[:nEvents],trainY[:nEvents]):
  xhat = np.concatenate((x,[1]))
  ds.addSample(xhat,y)

dims=[nRawVars+3,nRawVars+3,1] # one extra for constant offset
shallowNN = buildNetwork(dims[0],dims[1],dims[2])
# shallowNN = nn.Network(3,dims,.1)
trainer = BackpropTrainer(shallowNN, ds)
print "beginning training"
print trainer.train()

for x,y in zip(list(trainX)[:nEvents],trainY[:nEvents]):
  xhat = np.concatenate((x,[1]))
  yhat=shallowNN.activate(xhat)
  print y,yhat,abs(y-yhat)