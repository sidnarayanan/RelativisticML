#!/usr/bin/env python

import nn
import numpy as np
import ROOT
from sys import exit,stderr,stdout
from math import isnan

importWeights = False # True if you just want to load some weights (can either train further or just proceed)
trainWeights = True # if True and importWeights==True, then initializes with imported weights and trains further
printWeights =  True # saves weights in classifierStringWeights.py
classifierString="vtagSigmoidWeightedSmallData3Hidden"
print classifierString
# classifierString="vtagBestLargeData3Hidden"
negVal=0
largeDatasets=False

def validate(trainX,trainY,trainWeight,indices):
  nBins=1000
  sh=ROOT.TH1F("sht","sht",nBins,negVal,1)
  bh=ROOT.TH1F("bht","bht",nBins,negVal,1)
  for idx in indices:
    x=trainX[idx]
    y=trainY[idx]
    weight=trainWeight[idx]
    xhat = np.concatenate((x,[1]))
    yhat = shallowNN.evaluate(xhat)
    if y==1:
      sh.Fill(yhat,weight)
    else:
      bh.Fill(yhat,weight)
  nSig = float(sh.Integral())
  nBg = float(bh.Integral())
  for cut in range(1,nBins):
    nerrSig = sh.Integral(1,cut)/nSig
    if 0.49 <nerrSig < 0.51:
      nerrBg = bh.Integral(cut,nBins)/nBg
      return nerrBg

listOfRawVars = ["fjet1QGtagSub1","fjet1QGtagSub2","fjet1QGtag","fjet1PullAngle","fjet1Pull","fjet1MassTrimmed","fjet1MassPruned","fjet1MassSDbm1","fjet1MassSDb2","fjet1MassSDb0","fjet1QJetVol","fjet1C2b2","fjet1C2b1","fjet1C2b0p5","fjet1C2b0p2","fjet1C2b0","fjet1Tau2","fjet1Tau1"]   
# listOfRawVars = ["fjet1QGtagSub1","fjet1QGtagSub2","fjet1Tau2","fjet1Tau1"]   
nRawVars = len(listOfRawVars)
if not largeDatasets:
  sigFile = ROOT.TFile("../signal_word.root")
  bgFile = ROOT.TFile("../background_word.root")
  sigTree = sigFile.Get("DMSTree")
  bgTree = bgFile.Get("DMSTree")
else: # higher statistics but io is slow
  sigFile = ROOT.TFile("~/myscratch/ttlj_reduced_v2.root")
  bgFile = ROOT.TFile("~/myscratch/zll_reduced_v2.root")
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
sigY = []
bgX = np.empty([bgTree.GetEntries(),nRawVars+2]) # two extra computed
bgY = []
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
      sigY.append(1)
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
      bgY.append(negVal)
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
trainX = np.vstack((sigX,bgX))
mean=trainX.mean(0)
std=trainX.std(0)
# sigX = (sigX - mean)/std # center and normalize
# bgX = (bgX - mean)/std
trainX = (trainX - mean)/std
trainY = np.hstack((sigY,bgY))
trainWeight = np.hstack((sigWeight,bgWeight))
trainFrac = 3./4.
sigindices = np.array(range(sigcount))
bgindices = np.array(range(bgcount))
nTrainSig = int(sigcount*trainFrac)
nTrainBg = int(bgcount*trainFrac)
if not largeDatasets:
  maxSigTrainWeight=max(sigWeight[:nTrainSig])
  maxBgTrainWeight=max(bgWeight[:nTrainBg])
  sigTrainIndices = np.empty(max(nTrainSig,nTrainBg))
  bgTrainIndices = np.empty(max(nTrainSig,nTrainBg))
else:
  sigTrainIndices = sigindices[np.random.randint(0,nTrainSig,max(nTrainSig,nTrainBg))] # equal number of signal and bg events, choose with replacement
  bgTrainIndices = bgindices[np.random.randint(0,nTrainBg,max(nTrainSig,nTrainBg))] + sigcount # offset by number of signal events
  trainIndices = np.hstack((sigTrainIndices,bgTrainIndices))
testIndices = np.hstack((sigindices[nTrainSig:], bgindices[nTrainBg:]))
# dims=[nRawVars+3,nRawVars+3,1] # one extra for constant offset
dims=[nRawVars+3,nRawVars+3,nRawVars+3,nRawVars+3,1] # one extra for constant offset
shallowNN = nn.Network(5,dims,.1)

if importWeights:
  import vTagWeights
  shallowNN.weights = vTagWeights.w
if trainWeights:
  besterr=1
  i=0
  for nEpoch in range(25):
    print "Epoch ",nEpoch
    if not largeDatasets:
      for j in range(max(nTrainSig,nTrainBg)): # gibbs sample data according to weights
        good=False
        while not good:
          testidx = np.random.randint(0,nTrainSig)
          testidy = np.random.random()*maxSigTrainWeight
          if testidy < sigWeight[testidx]:
            sigTrainIndices[j] = testidx
            good=True
        good=False
        while not good:
          testidx = np.random.randint(0,nTrainBg)
          testidy = np.random.random()*maxBgTrainWeight
          if testidy < bgWeight[testidx]:
            bgTrainIndices[j] = testidx
            good=True
      trainIndices = np.hstack((sigTrainIndices,bgTrainIndices))
    print "Training on %i events"%(len(trainIndices))
    np.random.shuffle(trainIndices) # mix up signal and background
    for idx in trainIndices:
      x=trainX[idx]
      y=trainY[idx]
      # for x,y in zip(list(trainX)[:nEvents],trainY[:nEvents]):
      if not i%10000:
        # stderr.write("training %i\n"%(i))
        stdout.write("training %i\n"%(i))
      i+=1
      xhat = np.concatenate((x,[1]))
      yhat= shallowNN.evaluate(xhat)
      shallowNN.backProp(xhat,y)
      # if not i%100:
      if True:
        try:
          print yhat,y,abs(float(y)-yhat)
          # print "error ",abs(float(y)-yhat)
        except RuntimeWarning:
          # probably means something went out of bound in the network, no point continuing
          # this does not work fix it later
          print y,yhat
          exit(-1)
    validateErr=validate(trainX,trainY,trainWeight,testIndices)
    print "validation error ",validateErr
    if printWeights and validateErr<besterr:
        with open(classifierString+str(nEpoch)+'_Weights.py','w') as weightFile:
          weightFile.write("#!/usr/bin/env python\nfrom numpy import array\nw=")
          weightFile.write("%s\n"%(str(shallowNN.weights)))
        besterr=validateErr
  if printWeights:
    with open(classifierString+'Weights.py','w') as weightFile:
      weightFile.write("#!/usr/bin/env python\nfrom numpy import array\nw=")
      weightFile.write("%s\n"%(str(shallowNN.weights)))

nerr={}
nBins=1000
for cut in range(nBins):
  nerr[cut] = {negVal:0,1:0}
# for x,y in zip(list(trainX),validateY):
fout = ROOT.TFile('vlogs/'+classifierString+'.root','RECREATE')
sh=ROOT.TH1F("sh","sh",nBins,negVal,1)
bh=ROOT.TH1F("bh","bh",nBins,negVal,1)
for idx in testIndices:
  x=trainX[idx]
  y=trainY[idx]
  weight=trainWeights[idx]
  xhat = np.concatenate((x,[1]))
  yhat = shallowNN.evaluate(xhat)
  if y==1:
    sh.Fill(yhat,weight)
  else:
    bh.Fill(yhat,weight)
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
