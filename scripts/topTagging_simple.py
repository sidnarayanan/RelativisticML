#!/usr/bin/python

import cPickle as pickle
import numpy as np
from theano import config
import theano.tensor as T
import Classifiers.NeuralNet as NN
import ROOTInterface.Import
import ROOTInterface.Export
import sys
import ROOT as root # turned off to run on t3
from os import fsync

thingsToDo = 0 if len(sys.argv)==1 else int(sys.argv[1])

config.int_division = 'floatX'
def divide(a):
	return a[0]/a[1]
def bin(a,b,m):
	return min(int(a[0]/b),m)

lossFile = sys.stdout
msgFile = sys.stderr

rng = np.random.RandomState()
x = T.matrix('x')

# listOfRawVars = []
listOfRawVars = ["massSoftDrop","QGTag","maxSubjetBtag"]
listOfComputedVars = [(divide,['tau3','tau2']),(divide,['tau2','tau1'])]
nVars = len(listOfComputedVars) + len(listOfRawVars)

# dataPath = '/home/sid/scratch/data/topTagging_SDTopMass150/'
dataPath = '/home/sid/scratch/data/topTagging_SDTopWidth25/'

if thingsToDo&1:
	sigImporter = ROOTInterface.Import.TreeImporter(dataPath+'signal_AK8fj.root','jets')
	for v in listOfRawVars:
		sigImporter.addVar(v)
	for v in listOfComputedVars:
		sigImporter.addComputedVar(v)
	bgImporter = sigImporter.clone(dataPath+'qcd_AK8fj.root','jets')
	sigX,sigY = sigImporter.loadTree(1,-1)
	nSig = sigY.shape[0]
	bgX,bgY = bgImporter.loadTree(0,-1)
	nBg = bgY.shape[0]
	dataX = np.vstack([sigX,bgX])
	dataY = np.hstack([sigY,bgY])

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
	bgImporter.resetVars()
	def massBin(a):
		return bin(a,20,250)
	sigImporter.addVar('massSoftDrop')
	bgImporter.addVar('massSoftDrop')
	mass = np.vstack([sigImporter.loadTree(0,-1)[0],
					  bgImporter.loadTree(0,-1)[0]])
	massBinned = np.array([massBin(m) for m in mass])

	with open(dataPath+"compressed.pkl",'wb') as pklFile:
		pickle.dump({'nSig':nSig,  'nBg':nBg, 
									'dataX':dataX,
									'dataY':dataY,
									'mass':mass, # for plotting
								 	'massBinned':massBinned,
								 	'mu':mu,
								 	'sigma':sigma},pklFile,-1)

if thingsToDo&2:
	if not(thingsToDo&1):
		with open(dataPath+"compressed.pkl",'rb') as pklFile:
			d = pickle.load(pklFile)
			dataX = d['dataX']
			dataY = d['dataY']
			massBinned = d['massBinned']
			mass = d['mass']
			nSig = d['nSig']
			nBg = d['nBg']
	print dataX[:10]
	nData = dataY.shape[0]

	nTrain = nData*1/2-5000
	nTest = 10000
	nValidate = nData-nTrain-nTest
	# nValidate = nData*1/16
	learningRate = .01
	nSinceLastImprovement = 0
	bestTestLoss = np.inf
	sigTestLoss = np.inf
	epoch=0
	iteration=0
	nEpoch=1000
	patienceBaseVal = 10000 # do at least this many iterations
	patience = patienceBaseVal
	patienceFactor = 1.1
	significantImprovement = .995
	done=False
	nPerBatch=200

	classifier = NN.NeuralNet(x,rng,[nVars,nVars*5,nVars*5,nVars*5,nVars*5,nVars*5,nVars*5,nVars*5,2])
	# classifier.setSignalWeight(float(nBg)/nSig)
	trainer,loss = classifier.getTrainer(0,0,"NLL")
	print "Done with initialization!"

	dataIndices = np.arange(nData)
	np.random.shuffle(dataIndices) # mix up signal and background
	trainIndices = dataIndices[:nTrain]
	validateIndices = dataIndices[nTrain:nTrain+nValidate]
	testIndices = dataIndices[nTrain+nValidate:]
	# mask = np.ones(nTrain,dtype=bool)
	# for i in xrange(nTrain):
	# 	if mass[trainIndices[i]] > 120 or mass[trainIndices[i]] < 50:
	# 		mask[i] = False
	# trainIndices = trainIndices[mask]
	# nTrain = trainIndices.shape[0]
	msgFile.write("%d\n"%(nTrain))
	lossFile.write("%f\n"%(classifier.errors(dataX[testIndices],dataY[testIndices])))
	bestParameters = None

	print "Starting training!"
	while (epoch<nEpoch):
		np.random.shuffle(trainIndices) # randomize learning order
		msgFile.write("Epoch: %i\n"%(epoch))
		for i in xrange(nTrain/nPerBatch):
			if nSinceLastImprovement == 5:
				nSinceLastImprovement=0
				learningRate = learningRate*.1
				msgFile.write("\tLearningRate: %f\n"%(learningRate))
				classifier.initialize(bestParameters) # go back to the best point
			idx = trainIndices[i*nPerBatch:(i+1)*nPerBatch]
			trainer(dataX[idx],dataY[idx],learningRate)
			if not iteration%50:
				msgFile.write("Iteration: %i\n"%(iteration))
				testLoss = loss(dataX[testIndices],dataY[testIndices])[0]
				# testLoss = NN.evaluateZScore(classifier.probabilities(dataX[testIndices]),dataY[testIndices],None,False)
				# testLoss = classifier.errors(dataX[testIndices],dataY[testIndices])
				lossFile.write("%f\n"%(testLoss))
				if testLoss < bestTestLoss:
					nSinceLastImprovement=0
					msgFile.write("\tNewBestLoss: %f\n"%(testLoss))
					bestTestLoss = testLoss
					bestParameters = classifier.getParameters()
					# NN.evaluateZScore(classifier.probabilities(dataX[validateIndices]),dataY[validateIndices],mass[validateIndices],False)
					if testLoss/sigTestLoss < significantImprovement:
						patience = patienceBaseVal+iteration*patienceFactor
						msgFile.write("\tIncreasingPatience: %f\n"%(patience))
						sigTestLoss = testLoss
				else:
					nSinceLastImprovement+=1
			iteration+=1
			if iteration > patience:
			# if iteration > 1000:
				done=True
				break
			if learningRate <= 0.0000001:
				done = True
				break
		if done:
			break
		epoch+=1

	classifier.initialize(bestParameters)
	print NN.evaluateZScore(classifier.probabilities(dataX[validateIndices]),dataY[validateIndices],mass[validateIndices],True)

	with open("bestParams_simple.pkl",'wb') as pklFile:
		pickle.dump(bestParameters,pklFile,-1)

	with open("topTagger_simple.icc","w") as fOut:
		exporter = ROOTInterface.Export.NetworkExporter(classifier)
		exporter.setFile(fOut)
		exporter.export('topANN_simple')