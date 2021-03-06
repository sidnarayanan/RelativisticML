#!/usr/bin/env python

import nn
import numpy as np
import ROOT
from sys import exit,stderr,stdout,argv
from math import isnan
from os import getpid

print getpid() # to make it easier to kill

importClassifier = False # True to load data and weights
importWeights = False # False if you want to randomly initialize weights
largeDatasets = True # use high statistics samples but have slow i/o

negVal=0
listOfRawVars = ["fjet1QGtagSub1","fjet1QGtagSub2","fjet1QGtag","fjet1PullAngle","fjet1Pull","fjet1MassTrimmed","fjet1MassPruned","fjet1MassSDbm1","fjet1MassSDb2","fjet1MassSDb0","fjet1QJetVol","fjet1C2b2","fjet1C2b1","fjet1C2b0p5","fjet1C2b0p2","fjet1C2b0","fjet1Tau2","fjet1Tau1"]   
nRawVars = len(listOfRawVars)
nHidden=3
dims=[nRawVars+3]
for i in xrange(nHidden):
  dims.append(nRawVars+3)
dims.append(1)

dataX = None
dataY = None

def validate(trainX,trainY,indices):
  nBins=1000
  sh=ROOT.TH1F("sht","sht",nBins,negVal,1)
  bh=ROOT.TH1F("bht","bht",nBins,negVal,1)
  for idx in indices:
    x=trainX[idx]
    y=trainY[idx]
    xhat = np.concatenate((x,[1]))
    yhat = shallowNN.evaluate(xhat)
    if y==1:
      sh.Fill(yhat)
    else:
      bh.Fill(yhat)
  nSig = float(sh.Integral())
  nBg = float(bh.Integral())
  for margin in [(.49,.51), (.45,.55), (.4,.6)]:
    for cut in range(1,nBins):
      nerrSig = sh.Integral(1,cut)/nSig
      if margin[0] <nerrSig < margin[1]:
        nerrBg = bh.Integral(cut,nBins)/nBg
        return nerrBg

if not importClassifier:
  if not largeDatasets:
    sigFile = ROOT.TFile("~/scratch/signal_word.root")
    bgFile = ROOT.TFile("~/scratch/background_word.root")
    sigTree = sigFile.Get("DMSTree")
    bgTree = bgFile.Get("DMSTree")
  else: # higher statistics but io is slow
    sigFile = ROOT.TFile("~/scratch/ttlj_reduced_v2.root")
    bgFile = ROOT.TFile("~/scratch/zll_reduced_v2.root")
    sigTree = sigFile.Get("Events")
    bgTree = bgFile.Get("Events")
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
  sigX = np.empty([sigTree.GetEntries(),nRawVars+2]) # two extra computed
  sigY = np.empty([sigTree.GetEntries()])
  bgX = np.empty([bgTree.GetEntries(),nRawVars+2]) # two extra computed
  bgY = np.empty([bgTree.GetEntries()])
  sigcount=0
  bgcount=0
  if not largeDatasets:
    sigWeight = []
    bgWeight = []
  for n in range(sigTree.GetEntries()):
      sigTree.GetEntry(n)
      m=0
      if not largeDatasets: # selection cuts
        fjet1PartonId = sigDict["fjet1PartonId"].GetValue()
        if abs(fjet1PartonId)!=24:
          # not a w
          continue
      else:
        fjet1PartonId = sigDict["lJetId"].GetValue()
        passT = sigDict["lPassT"].GetValue()
        fjet1pt = sigDict["lJPt1"].GetValue()
        fjet1MassPruned = sigDict["fjet1MassPruned"].GetValue()
        fatjetid = sigDict["lFatJet"].GetValue()
        if not(passT>0 and fjet1pt>250 and fjet1MassPruned<120 and fatjetid<2 and abs(fjet1PartonId)==24):
          continue
      goodEvent=True
      for name in listOfRawVars:
        sigX[sigcount,m]=sigDict[name].GetValue()
        if isnan(sigX[sigcount,m]):
          print "WARNING, event ",n,name," is nan in signal, skipping!"
          goodEvent=False
          break
        m+=1
      if not goodEvent:
        continue
      else:
        sigX[sigcount,nRawVars] = 2*sigX[sigcount,1] + sigX[sigcount,0] # 2*fjet1QGtagSub2+fjet1QGtagSub1
        sigX[sigcount,nRawVars+1] = float(sigX[sigcount,nRawVars-2])/sigX[sigcount,nRawVars-1] # tau2/tau1
        sigY[sigcount] = 1
        if not largeDatasets:
          sigWeight.append(sigDict["weight"].GetValue())
        sigcount+=1
  for n in range(bgTree.GetEntries()):
      bgTree.GetEntry(n)
      m=0
      if not largeDatasets: # selection cuts
        pass
      else:
        passZ = bgDict["lPassZ"].GetValue()
        fjet1pt = bgDict["lJPt1"].GetValue()
        fjet1MassPruned = bgDict["fjet1MassPruned"].GetValue()
        fatjetid = bgDict["lFatJet"].GetValue()
        if not(passZ>3 and fjet1pt>250 and fjet1MassPruned<120 and fatjetid<2):
          continue
      goodEvent=True
      for name in listOfRawVars:
        bgX[bgcount,m]=bgDict[name].GetValue()
        if isnan(bgX[bgcount,m]):
          print "WARNING, event ",n,name," is nan in bg, skipping!"
          goodEvent=False
          break
        m+=1
      if not goodEvent:
        continue
      else:
        bgX[bgcount,nRawVars] = 2*bgX[bgcount,1] + bgX[bgcount,0] # 2*fjet1QGtagSub2+fjet1QGtagSub1
        bgX[bgcount,nRawVars+1] = float(bgX[bgcount,nRawVars-2])/bgX[bgcount,nRawVars-1] # tau2/tau1
        bgY[bgcount] = negVal
        if not largeDatasets:
          bgWeight.append(bgDict["weight"].GetValue())
        bgcount+=1
  if not largeDatasets:
    sigWeight=np.array(sigWeight)
    bgWeight=np.array(bgWeight)
  print sigcount,"/",sigTree.GetEntries()
  print bgcount,"/",bgTree.GetEntries()
  sigX = sigX[:sigcount]
  bgX = bgX[:bgcount] # only include those that passed
  sigY = sigY[:sigcount]
  bgY = bgY[:bgcount]
  dataX = np.vstack((sigX,bgX))
  mean=dataX.mean(0)
  std=dataX.std(0)
  dataX = (dataX - mean)/std
  dataY = np.hstack((sigY,bgY))
  indices = np.arange(bgcount+sigcount)
  sigFile.Close()
  bgFile.Close()
  sigindices = np.array(range(sigcount))
  bgindices = np.array(range(bgcount))
  trainFrac = 2./3
  nTrainSig = int(sigcount*trainFrac)
  nTrainBg = int(bgcount*trainFrac)
  sigTrainIndices = sigindices[np.random.randint(0,nTrainSig,max(nTrainSig,nTrainBg))] # equal number of signal and bg events, choose with replacement
  bgTrainIndices = bgindices[np.random.randint(0,nTrainBg,max(nTrainSig,nTrainBg))] + sigcount # offset by number of signal events
  trainIndices = np.hstack((sigTrainIndices,bgTrainIndices))
testIndices = np.hstack((sigindices[nTrainSig:], bgindices[nTrainBg:]))
classifier = None
if not importClassifier:
  classifier = nn.Network(dims,(dataX,dataY))
else:
  pickleJarName = argv[1]+".pkl" # maybe specify using a tag later
  classifier = nn.Network(0)
  classifier.loadFromPickle(pickleJarName)
classifier.debug = True
classifier.createSummaryFile("testBatch.log")
classifier.printSummary()
classifier.randomizeIndices()

if not importWeights:
  # we should train the classifier
  nData = classifier.dataY.shape[0]
  classifier.randomizeIndices(nData/3*2)
  # classifier.trainSGD()
  trainX = classifier.dataX
  trainY = classifier.dataY
  trainIndices = classifier.trainingIndices
  testIndices = classifier.testingIndices
  shallowNN = classifier
  besterr=1
  i=0
  for nEpoch in range(25):
    print "Epoch ",nEpoch
    print "Training on %i events"%(len(trainIndices))
    # np.random.shuffle(trainIndices) # mix up signal and background
    for idx in trainIndices:
      x=trainX[idx]
      y=trainY[idx]
      # for x,y in zip(list(trainX)[:nEvents],trainY[:nEvents]):
      if not i%10000:
        # stderr.write("training %i\n"%(i))
        stdout.write("training %i\n"%(i))
      i+=1
      # xhat = x
      xhat = np.concatenate((x,[1]))
      yhat= shallowNN.evaluate(xhat)
      shallowNN.backProp(xhat,y)
      if not i%10000:
      # if True:
        try:
          print yhat,y,abs(float(y)-yhat)
          print shallowNN.testNetwork()
          # print "error ",abs(float(y)-yhat)
        except RuntimeWarning:
          # probably means something went out of bound in the network, no point continuing
          # this does not work fix it later
          print y,yhat
          exit(-1)
    validateErr=validate(trainX,trainY,testIndices)
    print validateErr

validationIndices = classifier.validationIndices
nBins=100
hSig = ROOT.TH1F("hSig","hSig",nBins,negVal,1)
hBg = ROOT.TH1F("hBg","hBg",nBins,negVal,1)
for i in validationIndices:
  x = classifier.dataX[i]
  y = classifier.dataY[i]
  yhat = classifier.evaluate(x)
  if y==1:
    hSig.Fill(yhat)
  else:
    hBg.Fill(yhat)
rootFile = ROOT.TFile(argv[1]+".root","RECREATE")
rootFile.cd()
hSig.Write()
hBg.Write()
rootFile.Close()

rocFile = open(argv[1]+".roc",'w')
nSig = float(hSig.Integral())
nBg = float(hBg.Integral())
for cut in xrange(nBins):
  sigErr = hSig.Integral(1,cut)/nSig
  bgErr = hBg.Integral(cut,nBins)/nBg
  rocFile.write("%i %f %f\n"%(cut,1-sigErr,bgErr))
rocFile.close()

classifier.destroySummaryFile()
classifier.dumpToPickle(argv[1]+".pkl")
