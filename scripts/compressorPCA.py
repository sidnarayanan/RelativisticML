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

compressedName = 'compressedPCAWindow'
listOfRawVars = ["xformed_logchi","xformed_QGTag","xformed_QjetVol","xformed_groomedIso","xformed_sjqgtag0","xformed_sjqgtag1","xformed_sjqgtag2","xformed_tau32"]
listOfComputedVars = [] # third property is short name
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
    listOfCuts.append((lambda m: np.abs(m[0]-172.5) < 30., ['massSoftDrop']))
  else:
    listOfCuts.append((lambda m: np.abs(m[0]-172.5) < 20., ['massSoftDrop']))

compressedName += "_%i_%i_%.1f"%(int(ptLow),int(ptHigh),etaHigh)
compressedName = compressedName.replace('.','p')
print '%f < pT < %f && |eta| < %f, %s'%(ptLow,ptHigh,etaHigh,jetAlgo)


# dataPath = '/home/sid/scratch/data/topTagging_SDTopmass150/'
dataPath = '/home/snarayan/cms/root/topTagging_%s/'%(jetAlgo)
# dataPath = '/home/sid/scratch/data/topTagging_%s/'%(jetAlgo)

# first tagging variables
sigImporter = ROOTInterface.Import.TreeImporter(dataPath+'signal.root','jets')
for v in listOfRawVars:
  sigImporter.addVar(v)
for v in listOfComputedVars:
  sigImporter.addComputedVar(v)
for c in listOfCuts:
  sigImporter.addCut(c)
bgImporter = sigImporter.clone(dataPath+'qcd.root','jets')
sigImporter.addFriend('disc') # to get xformed variables
bgImporter.addFriend('disc')

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

V = np.empty([nVars,nVars])
with open(dataPath+'/pca_%i_%i_%s.pkl'%(ptLow,ptHigh,jetAlgo),'rb') as pcaFile:
  eigs = pickle.load(pcaFile)
  for i in xrange(nVars):
    a,v = eigs[i]
    print a
    V[:,i] = v

truncV = V[:,1:] # kill leading component
dataX = np.dot(dataX,truncV)
nVars -= 1

# longSuffix = ('_ptGT%.1fANDptLT%.1fANDabsetaLT%.1f'%(ptLow,ptHigh,etaHigh)).replace('.','p')
# alphas = np.empty(nVars)
# V = np.empty([nVars,nVars])
# with open(dataPath+'/pca.txt') as pcaFile:
#   for line in pcaFile:
#     if line.find(longSuffix) >= 0:
#       print line
#       ll = line.split()
#       if ll[0]=='alpha':
#         alphas[int(ll[1])] = float(ll[-1])
#       else:
#         for i in xrange(nVars):
#           # print i,ll[3+i]
#           V[i,int(ll[1])] = float(ll[3+i])

# truncV = V[:,1:] # kill leading component
# dataX = np.dot(dataX,truncV)
# nVars -= 1

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

# sigImporter.resetVars()
# bgImporter.resetVars()
# sigImporter.resetCounter()
# bgImporter.resetCounter()
# def massBin(a):
#   return bin(a,20,250)
# sigImporter.addVar('massSoftDrop')
# sigImporter.addVar('pt')
# sigImporter.addVar('eta')
# bgImporter.addVar('massSoftDrop')
# bgImporter.addVar('pt')
# bgImporter.addVar('eta')
# for c in listOfCuts:
#   sigImporter.addCut(c)
#   bgImporter.addCut(c)
# if doMultiThread:
#   sigKinematics = sigImporter.loadTreeMultithreaded(0,nEvents)[0]
#   bgKinematics = bgImporter.loadTreeMultithreaded(0,nEvents)[0]
#   kinematics = np.vstack([sigKinematics,bgKinematics])
# else:
#   sigKinematics = sigImporter.loadTree(0,nEvents)[0]
#   bgKinematics = bgImporter.loadTree(0,nEvents)[0]
#   kinematics = np.vstack([sigKinematics,bgKinematics])
# # massBinned = np.array([massBin([m]) for m in kinematics[:,0]])

# # bgImporter.resetVars()
# # bgImporter.resetCounter()
# # bgImporter.addFriend('weights')
# # bgImporter.addVar('weight')
# # for c in listOfCuts:
# #   bgImporter.addCut(c)
# # bgWeights = bgImporter.loadTree(0,nEvents)[0][:,0]
# # sigWeights = sigY
# # weights = np.hstack([sigWeights,bgWeights])
# # print sigWeights.shape,bgWeights.shape,weights.shape
# # massBinned = np.array([massBin([m]) for m in kinematics[:,0]])

# print 'finished loading %i kinematics'%(kinematics.shape[0])

# print kinematics[:10]

# sigImporter = ROOTInterface.Import.TreeImporter(dataPath+'signal_weights_CA15fj.root','weights')
# bgImporter = ROOTInterface.Import.TreeImporter(dataPath+'qcd_weights_CA15fj.root','weights')
# sigImporter.addVar('weight')
# bgImporter.addVar('weight')
# weight = np.vstack([sigImporter.loadTree(0,nEvents)[0]*nBg,
#             bgImporter.loadTree(0,nEvents)[0]]*nSig)


with open(dataPath+compressedName+".pkl",'wb') as pklFile:
  pickle.dump({'nSig':nSig,  'nBg':nBg, 
                'dataX':dataX,
                'dataY':dataY,
                # 'kinematics':kinematics, # for plotting
                # 'weights':weights,
                # 'massBinned':massBinned,
                'mu':mu,
                'sigma':sigma,
                'vars':listOfRawVarsNames},pklFile,-1)

with open(dataPath+compressedName+"_small.pkl",'wb') as pklFile:
  pickle.dump({'nSig':nSig,  'nBg':nBg, 
                'mu':mu,
                'sigma':sigma,
                'vars':listOfRawVarsNames},pklFile,-1)

print 'done!'
