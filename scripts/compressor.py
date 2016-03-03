#!/usr/bin/python

import cPickle as pickle
import numpy as np
import ROOTInterface.Import
import ROOTInterface.Export
from sys import argv

nEvents = -1
doMultiThread = False

def divide(a):
  return a[0]/a[1]
def bin(a,b,m):
  return min(int(a[0]/b),m)
def angleTruncate(a):
  return min(6.28,max(0,a[0]))

print "starting!"

rng = np.random.RandomState()

compressedName = 'compressedBasic'
#listOfRawVars = ["massPruned","massTrimmed","MassSDb0","MassSDb1","MassSDb2","MassSDbm1","logchi","QGTag","QjetVol","groomedIso","sjqgtag0","sjqgtag1","sjqgtag2"]
#listOfRawVars = ["logchi","QGTag","QjetVol","groomedIso","sjqgtag0","sjqgtag1","sjqgtag2"]
#listOfRawVars = ["logchi","QjetVol","groomedIso","dR_sj0dR","dR_sj0mW","nSubjets","dR_sj0dPhiTRF"]
listOfRawVars = ["logchi","QjetVol","groomedIso","nSubjets"]
listOfComputedVars = [(divide,['tau3','tau2'],'tau32')] # third property is short name
# for i in range(6):
#   listOfRawVars.append('iota%i'%(i))
#   listOfComputedVars.append((angleTruncate,['iotaAngle%i'%(i)],'iotaAngle%i'%(i)))
listOfCuts = []
nVars = len(listOfComputedVars) + len(listOfRawVars)
listOfRawVarsNames = []
for v in listOfRawVars:
  listOfRawVarsNames.append(v)
for f,v,n in listOfComputedVars:
  listOfRawVarsNames.append(n)

if len(argv)>1:
  ptLow = float(argv[1])
  ptHigh = float(argv[2])
  etaHigh = float(argv[3])
  jetAlgo = argv[4]
  listOfCuts.append((lambda eta: np.abs(eta[0]) < etaHigh, ['eta']))
  listOfCuts.append((lambda pt: pt[0] > ptLow, ['pt']))
  listOfCuts.append((lambda pt: pt[0] < ptHigh, ['pt']))
  if jetAlgo=='CA15':
    listOfCuts.append((lambda m: 150<m[0] and m[0]<240, ['massSoftDrop']))
  else:
    listOfCuts.append((lambda m: 110<m[0] and m[0]<210, ['massSoftDrop']))

compressedName += "_%i_%i_%.1f"%(int(ptLow),int(ptHigh),etaHigh)
compressedName = compressedName.replace('.','p')
print '%f < pT < %f && |eta| < %f, %s'%(ptLow,ptHigh,etaHigh,jetAlgo)


dataPath = '/home/snarayan/cms/root/topTagging_%s/'%(jetAlgo)

# first tagging variables
sigImporter = ROOTInterface.Import.TreeImporter(dataPath+'signal.root','jets')
for v in listOfRawVars:
  sigImporter.addVar(v)
for v in listOfComputedVars:
  sigImporter.addComputedVar(v)
for c in listOfCuts:
  sigImporter.addCut(c)
bgImporter = sigImporter.clone(dataPath+'qcd.root','jets')

print "finished setting up TreeImporters"

sigX,sigY = sigImporter.loadTree(1,nEvents)
print sigY
nSig = sigY.shape[0]
print '\tloaded %i signal'%(nSig)
bgX,bgY = bgImporter.loadTree(0,nEvents)
nBg = bgY.shape[0]
print '\tloaded %i background'%(nBg)
dataX = np.vstack([sigX,bgX])
dataY = np.hstack([sigY,bgY])

print 'finished loading dataX and dataY: %i events'%(dataY.shape[0])

mu = dataX.mean(0)
sigma = dataX.std(0)
for i in xrange(sigma.shape[0]):
  # for constant rows, do not scale
  if not sigma[i]:
    sigma[i] = 1
    mu[i] = 0
dataX = (dataX - mu)/sigma

print "sample mu:",mu
print "sample sigma:",sigma

sigImporter.resetVars()
sigImporter.resetCounter()
sigImporter.addVar('mcWeight')
bgImporter.resetVars()
bgImporter.resetCounter()
bgImporter.addVar('mcWeight')
sigImporter.useGoodEntries = True
bgImporter.useGoodEntries = True
for c in listOfCuts:
  bgImporter.addCut(c)
  sigImporter.addCut(c)
sigWeights = sigImporter.loadTree(0,nEvents)[0][:,0]
print "loaded %i signal weights"%(sigWeights.shape[0])
bgWeights = bgImporter.loadTree(0,nEvents)[0][:,0]
print "loaded %i background weights"%(bgWeights.shape[0])

sigTotal = sigWeights.sum()
bgTotal = bgWeights.sum()
sigWeights *= bgTotal/(sigTotal+bgTotal)
bgWeights *= sigTotal/(sigTotal+bgTotal)

weights = np.hstack([sigWeights,bgWeights])

with open(dataPath+compressedName+".pkl",'wb') as pklFile:
  pickle.dump({'nSig':nSig,  'nBg':nBg, 
                'dataX':dataX,
                'dataY':dataY,
                'weights':weights,
                'mu':mu,
                'sigma':sigma,
                'vars':listOfRawVarsNames},pklFile,-1)

with open(dataPath+compressedName+"_small.pkl",'wb') as pklFile:
  pickle.dump({'nSig':nSig,  'nBg':nBg, 
                'mu':mu,
                'sigma':sigma,
                'vars':listOfRawVarsNames},pklFile,-1)

print 'done!'
