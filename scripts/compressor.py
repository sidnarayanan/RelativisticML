#!/usr/bin/python

import cPickle as pickle
import numpy as np
import ROOTInterface.Import
import ROOTInterface.Export
# import sys
# import ROOT as root # not need for compressor
# from os import fsync

nEvents = -1
doMultiThread = False

def divide(a):
	return a[0]/a[1]
def bin(a,b,m):
	return min(int(a[0]/b),m)

print "starting!"

rng = np.random.RandomState()

# listOfRawVars = []
listOfRawVars = ["massSoftDrop","QGTag","QjetVol","groomedIso"]
listOfComputedVars = [(divide,['tau3','tau2'])]
nVars = len(listOfComputedVars) + len(listOfRawVars)

# dataPath = '/home/sid/scratch/data/topTagging_SDTopMass150/'
dataPath = '/home/snarayan/cms/root/topTagging_CA15/'
# dataPath = '/home/sid/scratch/data/ak8fj/'

# first tagging variables
sigImporter = ROOTInterface.Import.TreeImporter(dataPath+'signal_CA15fj.root','jets')
for v in listOfRawVars:
	sigImporter.addVar(v)
for v in listOfComputedVars:
	sigImporter.addComputedVar(v)
bgImporter = sigImporter.clone(dataPath+'qcd_CA15fj.root','jets')

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

print 'finished loading dataX and dataY'

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

sigImporter.resetVars()
bgImporter.resetVars()
def massBin(a):
	return bin(a,20,250)
sigImporter.addVar('massSoftDrop')
sigImporter.addVar('pt')
sigImporter.addVar('eta')
bgImporter.addVar('massSoftDrop')
bgImporter.addVar('pt')
bgImporter.addVar('eta')
if doMultiThread:
	kinematics = np.vstack([sigImporter.loadTreeMultithreaded(0,nEvents)[0],
						  bgImporter.loadTreeMultithreaded(0,nEvents)][0])
else:
	kinematics = np.vstack([sigImporter.loadTree(0,nEvents)[0],
						  bgImporter.loadTree(0,nEvents)][0])
# massBinned = np.array([massBin([m]) for m in kinematics[:,0]])

print 'finished loading kinematics'

# sigImporter = ROOTInterface.Import.TreeImporter(dataPath+'signal_weights_CA15fj.root','weights')
# bgImporter = ROOTInterface.Import.TreeImporter(dataPath+'qcd_weights_CA15fj.root','weights')
# sigImporter.addVar('weight')
# bgImporter.addVar('weight')
# weight = np.vstack([sigImporter.loadTree(0,nEvents)[0]*nBg,
# 				  	bgImporter.loadTree(0,nEvents)[0]]*nSig)

with open(dataPath+"compressed.pkl",'wb') as pklFile:
	pickle.dump({'nSig':nSig,  'nBg':nBg, 
								'dataX':dataX,
								'dataY':dataY,
								'kinematics':kinematics, # for plotting
							 	# 'massBinned':massBinned,
							 	'mu':mu,
							 	'sigma':sigma},pklFile,-1)
print 'done!'
