#!/usr/bin/env python

import nn
import numpy as np
import ROOT
from sys import exit,stderr,stdout
from math import isnan

importWeights = False # True if you just want to load some weights (can either train further or just proceed)
trainWeights = True # if True and importWeights==True, then initializes with imported weights and trains further
printWeights =  False # saves weights in classifierStringWeights.py

classifierString="test"
print classifierString
negVal=0
listOfRawVars = ["fjet1QGtagSub1","fjet1QGtagSub2","fjet1QGtag","fjet1PullAngle","fjet1Pull","fjet1MassTrimmed","fjet1MassPruned","fjet1MassSDbm1","fjet1MassSDb2","fjet1MassSDb0","fjet1QJetVol","fjet1C2b2","fjet1C2b1","fjet1C2b0p5","fjet1C2b0p2","fjet1C2b0","fjet1Tau2","fjet1Tau1"]   
nRawVars = len(listOfRawVars)
# dims=[nRawVars+3,nRawVars+3,1] # one extra for constant offset
dims=[nRawVars+3,nRawVars+3,nRawVars+3,nRawVars+3,1] # one extra for constant offset

def validate(trainX,trainY,indices):
  nBins=1000
  sh=ROOT.TH1F("sht","sht",nBins,negVal,1)
  bh=ROOT.TH1F("bht","bht",nBins,negVal,1)
  for idx in indices:
    x=trainX[idx]
    y=trainY[idx]
    yhat = shallowNN.evaluate(x)
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


sigFile = ROOT.TFile("~/scratch/signal_word.root")
bgFile = ROOT.TFile("~/scratch/background_word.root")
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

sigX = np.empty([sigTree.GetEntries(),nRawVars+2]) # two extra computed
sigY = []
bgX = np.empty([bgTree.GetEntries(),nRawVars+2]) # two extra computed
bgY = []
sigcount=0
bgcount=0
for n in range(sigTree.GetEntries()):
    sigTree.GetEntry(n)
    m=0
    fjet1PartonId = sigDict["fjet1PartonId"].GetValue()
    if abs(fjet1PartonId)!=24:
      # not a w
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
      sigcount+=1
for n in range(bgTree.GetEntries()):
    bgTree.GetEntry(n)
    m=0
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
      bgcount+=1

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
trainFrac = 3./4.
sigindices = np.array(range(sigcount))
bgindices = np.array(range(bgcount))
nTrainSig = int(sigcount*trainFrac)
nTrainBg = int(bgcount*trainFrac)
sigTrainIndices = sigindices[np.random.randint(0,nTrainSig,max(nTrainSig,nTrainBg))] # equal number of signal and bg events, choose with replacement
bgTrainIndices = bgindices[np.random.randint(0,nTrainBg,max(nTrainSig,nTrainBg))] + sigcount # offset by number of signal events
trainIndices = np.hstack((sigTrainIndices,bgTrainIndices))
testIndices = np.hstack((sigindices[nTrainSig:], bgindices[nTrainBg:]))

shallowNN = nn.Network(dims,(trainX,trainY),.5)
print shallowNN
if importWeights:
  import vTagWeights
  shallowNN.weights = vTagWeights.w
if trainWeights:
  besterr=1
  i=0
  for nEpoch in range(25):
    print "Epoch ",nEpoch
    print "Training on %i events"%(len(trainIndices))
    np.random.shuffle(trainIndices) # mix up signal and background
    for idx in trainIndices:
      x=trainX[idx]
      y=trainY[idx]
      if not i%10000:
        # stderr.write("training %i\n"%(i))
        stdout.write("training %i\n"%(i))
      i+=1
      yhat= shallowNN.evaluate(x)
      shallowNN.backProp(x,y)
      if not i%10000:
      # if True:
        try:
          print yhat,y,abs(float(y)-yhat)
          # print "error ",abs(float(y)-yhat)
        except RuntimeWarning:
          # probably means something went out of bound in the network, no point continuing
          # this does not work fix it later
          print y,yhat
          exit(-1)
    validateErr=validate(trainX,trainY,testIndices)
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
print validate(trainX,trainY,testIndices)
for cut in range(nBins):
  nerr[cut] = {negVal:0,1:0}
# for x,y in zip(list(trainX),validateY):
fout = ROOT.TFile('vlogs/'+classifierString+'.root','RECREATE')
sh=ROOT.TH1F("sh","sh",nBins,negVal,1)
bh=ROOT.TH1F("bh","bh",nBins,negVal,1)
for idx in testIndices:
  x=trainX[idx]
  y=trainY[idx]
  yhat = shallowNN.evaluate(x)
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

sigFile.Close()
bgFile.Close()
