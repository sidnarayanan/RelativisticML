#!/usr/bin/python

import cPickle as pickle
import numpy as np
import ROOTInterface.Import
import ROOTInterface.Export
from sys import argv,exit

nEvents = -1
doMultiThread = False

def divide(a):
  return a[0]/a[1]
def bin(a,b,m):
  return min(int(a[0]/b),m)
def angleTruncate(a):
  return min(6.28,max(0,a[0]))

print "starting!"

listOfCuts = []
rng = np.random.RandomState()
if len(argv)>1:
  ptLow = float(argv[1])
  ptHigh = float(argv[2])
  etaHigh = float(argv[3])
  jetAlgo = argv[4]
  listOfCuts.append((lambda iE: iE[0]%2==0, ['eventNumber']))
  listOfCuts.append((lambda eta: np.abs(eta[0]) < etaHigh, [jetAlgo+'fj1_eta']))
  listOfCuts.append((lambda pt: pt[0] > ptLow, [jetAlgo+'fj1_pt']))
  listOfCuts.append((lambda pt: pt[0] < ptHigh, [jetAlgo+'fj1_pt']))
  listOfCuts.append((lambda n: n[0] > 0, ['n'+jetAlgo+'fj']))
  if jetAlgo=='CA15':
    listOfCuts.append((lambda m: m[0]>150 and m[0]<240, [jetAlgo+'fj1_mSD']))
  else:
    listOfCuts.append((lambda m: m[0]>110 and m[0]<210, [jetAlgo+'fj1_mSD']))


compressedName = 'compressedWindow'
listOfRawVars = [jetAlgo+'fj1_'+x for x in ["logchi","QGTag","QjetVol","groomedIso","sjqgtag0","sjqgtag1","sjqgtag2"]]
listOfComputedVars = [(divide,['CA15fj1_tau3','CA15fj1_tau2'],'tau32')] # third property is short name
nVars = len(listOfComputedVars) + len(listOfRawVars)
listOfRawVarsNames = []
for v in listOfRawVars:
  listOfRawVarsNames.append(v)
for f,v,n in listOfComputedVars:
  listOfRawVarsNames.append(n)

compressedName += "_%i_%i_%.1f"%(int(ptLow),int(ptHigh),etaHigh)
compressedName = compressedName.replace('.','p')
print '%f < pT < %f && |eta| < %f, %s'%(ptLow,ptHigh,etaHigh,jetAlgo)

dataPath = '/home/snarayan/cms/root/monotop25ns_v2/topTagging/'

# first tagging variables
sigImporter = ROOTInterface.Import.TreeImporter(dataPath+'signal.root','events')
for v in listOfRawVars:
  sigImporter.addVar(v)
for v in listOfComputedVars:
  sigImporter.addComputedVar(v)
for c in listOfCuts:
  sigImporter.addCut(c)
bgImporter = sigImporter.clone(dataPath+'qcd.root','events')
#sigImporter.addCut((lambda m: m[0]==1, [jetAlgo+'fj1_isMatched']))
print "finished setting up TreeImporters"

if doMultiThread:
  sigX,sigY = sigImporter.loadTreeMultithreaded(1,nEvents)
else:
  sigX,sigY = sigImporter.loadTree(1,nEvents)
nSig = sigY.shape[0]
print '\tloaded %i signal'%(nSig)
if doMultiThread:
  bgX,bgY = bgImporter.loadTreeMultithreaded(0,nEvents)
else:
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

# now kinematic variables - mass, pt, eta, weight(?)

bgImporter.resetVars()
bgImporter.resetCounter()
bgImporter.addVar('mcWeight')
sigImporter.resetVars()
sigImporter.resetCounter()
sigImporter.addVar('mcWeight')
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
