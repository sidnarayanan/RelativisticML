#!/usr/bin/env python

import bdt
import numpy as np
import ROOT
from sys import exit,stderr,stdout
from math import isnan

nTrees=400
classifierString = "vtagBDS"+str(nTrees)
print classifierString
negVal=0

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
trainWeights = np.zeros([nEvents])
sigcount=0
bgcount=0
for n in range(nEvents):
  #if not n%2: 
  if not n%2: 
    goodEvent=False
    while not goodEvent:
      sigTree.GetEntry(sigcount)
      sigcount+=1
      m=0
      goodEvent=True
      fjet1PartonId = sigDict["fjet1PartonId"].GetValue()
      if abs(fjet1PartonId)!=24:
        goodEvent=False
        # not a w
        continue
      for name in listOfRawVars:
        trainX[n,m]=sigDict[name].GetValue()
        if isnan(trainX[n,m]):
          print "WARNING, event ",sigcount,name," is nan in signal, skipping!"
          goodEvent=False
          break
        m+=1
      trainWeights[n]=sigDict["weight"].GetValue()
    trainY[n]=1
  else:
    goodEvent=False
    while not goodEvent:
      bgTree.GetEntry(bgcount)
      bgcount+=1
      m=0
      goodEvent=True
      for name in listOfRawVars:
        trainX[n,m]=bgDict[name].GetValue()
        if isnan(trainX[n,m]):
          print "WARNING, event ",bgcount,name," is nan in bg, skipping!"
          goodEvent=False
          break
        m+=1
      trainWeights[n]=bgDict["weight"].GetValue()
  trainX[n,nRawVars] = 2*trainX[n,1] + trainX[n,0] # 2*fjet1QGtagSub2+fjet1QGtagSub1
  trainX[n,nRawVars+1] = float(trainX[n,nRawVars-2])/trainX[n,nRawVars-1] # tau2/tau1
trainX = (trainX - trainX.mean(0))/trainX.std(0) # center and normalize
print trainX[0]
print trainY[0]

dim=nRawVars+2
forest = bdt.Forest(dim,nTrees,trainX[:nEvents],None)
# forest = bdt.Forest(dim,4,trainX[:nEvents],trainWeights[:nEvents])
forest.adaBoost(trainX,trainY)

nEvents = nEvents/2
validateX = np.empty([nEvents,nRawVars+2]) # two extra computed
validateY= np.zeros([nEvents])
validateWeights=np.zeros([nEvents])
bgcount=0
for n in range(nEvents):
  #if not n%2: 
  if not n%2: 
    goodEvent=False
    while not goodEvent:
      sigTree.GetEntry(sigcount)
      sigcount+=1
      m=0
      goodEvent=True
      fjet1PartonId = sigDict["fjet1PartonId"].GetValue()
      if abs(fjet1PartonId)!=24:
        goodEvent=False
        # not a w
        continue
      for name in listOfRawVars:
        validateX[n,m]=sigDict[name].GetValue()
        if isnan(validateX[n,m]):
          print "WARNING, event ",sigcount,name," is nan in signal, skipping!"
          goodEvent=False
          break
        m+=1
      validateWeights[n]=sigDict["weight"].GetValue()
    validateY[n]=1
  else:
    goodEvent=False
    while not goodEvent:
      bgTree.GetEntry(bgcount)
      bgcount+=1
      m=0
      goodEvent=True
      for name in listOfRawVars:
        validateX[n,m]=bgDict[name].GetValue()
        if isnan(validateX[n,m]):
          print "WARNING, event ",bgcount,name," is nan in bg, skipping!"
          goodEvent=False
          break
        m+=1
      validateWeights[n]=bgDict["weight"].GetValue()
  validateX[n,nRawVars] = 2*validateX[n,1] + validateX[n,0] # 2*fjet1QGtagSub2+fjet1QGtagSub1
  if validateX[n,nRawVars-1]==0:
    print n,validateX[n]
    exit(-1)
  validateX[n,nRawVars+1] = float(validateX[n,nRawVars-2])/validateX[n,nRawVars-1] # tau2/tau1
validateX = (validateX - validateX.mean(0))/validateX.std(0) # center and normalize


i=0

print validateX

nerr={}
nBins=1000
for cut in range(nBins):
  nerr[cut] = {negVal:0,1:0}
fout = ROOT.TFile('vlogs/'+classifierString+'.root','RECREATE')
sh=ROOT.TH1F("sh","sh",nBins,negVal,1)
bh=ROOT.TH1F("bh","bh",nBins,negVal,1)
for x,y in zip(validateX,validateY):
  i+=1
  xhat = x
  yhat = forest.evaluate(xhat)
  if y==1:
    sh.Fill(yhat)
  else:
    bh.Fill(yhat)
nSig = float(sh.Integral())
nBg = float(bh.Integral())
for cut in range(1,nBins):
  nerr[cut][1] = sh.Integral(1,cut)/nSig
  nerr[cut][negVal] = bh.Integral(cut,nBins)/nBg

sh.Write()
bh.Write()
fout.Close()

with open('vlogs/'+classifierString+'ROC.log','w') as logFile:
  for cut in range(nBins):
    logFile.write("%f %f %f\n"%(cut, 1.-nerr[cut][1], float(nerr[cut][negVal])))


# print forest

bgFile.Close()
sigFile.Close()

# print "VALIDATION SUMMARY:"
# print "classification error 0 %.3f"%(nerr[0]/ntot[0])
# print "classification error 1 %.3f"%(nerr[1]/ntot[1])
