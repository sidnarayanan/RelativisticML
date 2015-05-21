#!/usr/bin/env python

import numpy as np
import ROOT
from sys import exit,stderr,stdout
from math import isnan
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

loadClassifier=False
saveClassifier=True

listOfRawVars = ["fjet1QGtagSub1","fjet1QGtagSub2","fjet1QGtag","fjet1PullAngle","fjet1Pull","fjet1MassTrimmed","fjet1MassPruned","fjet1MassSDbm1","fjet1MassSDb2","fjet1MassSDb0","fjet1QJetVol","fjet1C2b2","fjet1C2b1","fjet1C2b0p5","fjet1C2b0p2","fjet1C2b0","fjet1Tau2","fjet1Tau1"]   
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
nEvents = 20000000
nEvents = min(nEvents,sigTree.GetEntries(),bgTree.GetEntries())
# nEvents = 2*(sigTree.GetEntries()+bgTree.GetEntries())/3
# use nEvents/2 for training, nEvents/2 for validation (from each sample)
print "Training on %i events"%(nEvents)
trainX = np.empty([nEvents,nRawVars+2]) # two extra computed
trainY= np.ones([nEvents])
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
    trainY[n]=-1
  trainX[n,nRawVars] = 2*trainX[n,1] + trainX[n,0] # 2*fjet1QGtagSub2+fjet1QGtagSub1
  trainX[n,nRawVars+1] = float(trainX[n,nRawVars-2])/trainX[n,nRawVars-1] # tau2/tau1

trainX = (trainX - trainX.mean(0))/trainX.std(0) # center and normalize
dim=nRawVars+2

if not loadClassifier:
  print "beginning training"
  bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=9),
                           algorithm="SAMME",
                           n_estimators=500)
  bdt.fit(trainX,trainY)
else:
  bdt = joblib.load('vTagBDT_scikit_depth9.pkl')

if saveClassifier:
  joblib.dump(bdt,'vTagBDT_scikit_depth9.pkl',compress=9)

print "done training!"
nEvents = nEvents/2
print "validating on %i events"%(nEvents)
validateX = np.empty([nEvents,nRawVars+2]) # two extra computed
validateY= np.ones([nEvents])
for n in range(nEvents):
  if not n%2: 
    goodEvent=False
    while not goodEvent:
      sigTree.GetEntry(sigcount)
      m=0
      goodEvent=True
      for name in listOfRawVars:
        validateX[n,m]=sigDict[name].GetValue()
        if isnan(validateX[n,m]):
          print "WARNING, in event ",sigcount,name," is nan in signal, skipping!"
          goodEvent=False
          break
        m+=1
      sigcount+=1
    validateY[n]=1
  else:
    goodEvent=False
    while not goodEvent:
      bgTree.GetEntry(bgcount)
      m=0
      goodEvent=True
      for name in listOfRawVars:
        validateX[n,m]=bgDict[name].GetValue()
        if isnan(validateX[n,m]):
          print "WARNING, in event ",bgcount,name," is nan in bg, skipping!"
          goodEvent=False
          break
        m+=1
      bgcount+=1
    validateY[n]=-1
  validateX[n,nRawVars] = 2*validateX[n,1] + validateX[n,0] # 2*fjet1QGtagSub2+fjet1QGtagSub1
  validateX[n,nRawVars+1] = float(validateX[n,nRawVars-2])/validateX[n,nRawVars-1] # tau2/tau1
validateX = (validateX - validateX.mean(0))/validateX.std(0) # center and normalize
i=0

nerr={}
ntot= {-1:0,1:0}
for i in range(100):
  nerr[i*0.02-1] = {-1:0,1:0}
with open('vtag_scikit_scores.log','w') as scoreFile:
  for x,y in zip(validateX,validateY):
    i+=1
    xhat = x
    yhat = bdt.decision_function(xhat)
    scoreFile.write("%i %f\n"%(y,yhat))
    ntot[y]+=1
    for i in range(100):
      classifyAs = 1 if yhat > i*0.02-1 else -1
      if not classifyAs==y:
        nerr[i*0.02-1][y]+=1

for i in range(100):
  cut=i*0.02-1
  stdout.write("%f %f %f\n"%( cut, 1.-float(nerr[cut][1])/ntot[1], float(nerr[cut][-1])/ntot[-1]))

# print forest

# print "VALIDATION SUMMARY:"
# print "classification error 0 %.3f"%(nerr[0]/ntot[0])
# print "classification error 1 %.3f"%(nerr[1]/ntot[1])
