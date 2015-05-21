#!/usr/bin/env python

import nn
import numpy as np
from sys import exit,stderr
import ROOT

importWeights = True
printWeights =  False
negVal=-1
classifierString="higgs3Hidden"
print classifierString

def validate(trainX,trainY):
  nBins=1000
  sh=ROOT.TH1F("sht","sht",nBins,negVal,1)
  bh=ROOT.TH1F("bht","bht",nBins,negVal,1)
  for x,y in zip(trainX,trainY):
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

nSkip = 1 # > 1 just for development use
trainYString = np.loadtxt("../training.csv",usecols=[32],skiprows=nSkip,dtype={'names': ('label',),'formats': ('S1',)},delimiter=',')
trainX = np.loadtxt("../training.csv",usecols=range(1,30),delimiter=',',skiprows=nSkip)
trainY = np.array(map( lambda y : 1 if y[0]=='s' else negVal , trainYString))
trainX = (trainX - trainX.mean(0))/trainX.std(0)

dims=[30,30,30,30,1] # one extra for constant offset
shallowNN = nn.Network(len(dims),dims,.001)
print shallowNN

i=0
nTrain = 200000
if not importWeights:
  besterr=1
  for nEpoch in range(15):
    print "Epoch ",nEpoch
    for x,y in zip(trainX[:nTrain],trainY[:nTrain]):
      if not i%10000:
        print "training ",i
      i+=1
      xhat = np.concatenate((x,[1]))
      yhat= shallowNN.evaluate(xhat)
      #yhat= 1 if shallowNN.evaluate(xhat) > 0.5 else 0
      shallowNN.backProp(xhat,y)
      print "|",yhat,"-",y,"|=",abs(float(y)-yhat)
    validateErr=validate(trainX[:nTrain],trainY[:nTrain])
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
if importWeights: 
  array=np.array
  print "importing weights"
  import weights
  shallowNN.weights = weights.w

j=0
nerr={}
nBins=1000
for cut in range(nBins):
  nerr[cut] = {negVal:0,1:0}
fout = ROOT.TFile('logs/'+classifierString+'.root','RECREATE')
sh=ROOT.TH1F("sh","sh",nBins,negVal,1)
bh=ROOT.TH1F("bh","bh",nBins,negVal,1)
for x,y in zip(trainX[nTrain:],trainY[nTrain:]):
  xhat = np.concatenate((x,[1]))
  yhat = shallowNN.evaluate(xhat)
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

with open('logs/'+classifierString+'ROC.log','w') as logFile:
  for cut in range(nBins):
    logFile.write("%f %f %f\n"%(cut, 1.-nerr[cut][1], float(nerr[cut][negVal])))

# with open('logs/'+classifierString+'Scores.log','w') as scoreFile:
#   for y,score in zip(trainY[nTrain:],scores):
#     scoreFile.write("%i %f\n"%(y,score))

# print "VALIDATION SUMMARY:"
# print "classification error 0 %.3f"%(nerr[0]/ntot[0])
# print "classification error 1 %.3f"%(nerr[1]/ntot[1])
