#!/usr/bin/python

import cPickle as pickle
import numpy as np
from theano import config
import theano.tensor as T
import Classifiers.NeuralNet as NN
import ROOTInterface.Import
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

listOfRawVars = ["QGTag","telescopingIso","groomedIso"]
listOfComputedVars = [(divide,['tau3','tau2']),
						(divide,['tau2','tau1']),
						(np.log,['chi'])]
# listOfRawVars = ["iota0","iota1","iota2","iota3","iota4"]
# listOfComputedVars = []
nVars = len(listOfComputedVars) + len(listOfRawVars)

dataPath = '/home/sid/scratch/data/topTagging_SDTopMass150/'

if thingsToDo&1:
	sigImporter = ROOTInterface.Import.TreeImporter(dataPath+'signal_AK8fj.root','jets')
	sigImporter.addVarList(listOfRawVars)
	for v in listOfComputedVars:
		sigImporter.addComputedVar(v)
	bgImporter = sigImporter.clone(dataPath+'qcd_AK8fj.root','jets')
	sigX,sigY = sigImporter.loadTree(1,-1)
	bgX,bgY = bgImporter.loadTree(0,-1)
	dataX = np.vstack([sigX,bgX])
	dataY = np.hstack([sigY,bgY])

	mu = dataX.mean(0)
	sigma = dataX.std(0)
	for i in xrange(sigma.shape[0]):
		# for constant rows, do not offset
		if not sigma[i]:
			sigma[i] = 1
			mu[i] = 0
	dataX = (dataX - mu)/sigma


	sigImporter.resetVars()
	bgImporter.resetVars()
	def massBin(a):
		return bin(a,20,250)
	sigImporter.addVarList(['massSoftDrop'])
	bgImporter.addVarList(['massSoftDrop'])
	mass = np.vstack([sigImporter.loadTree(0,-1)[0],
					  bgImporter.loadTree(0,-1)[0]])
	massBinned = np.array([massBin(m) for m in mass])

	with open(dataPath+"compressed.pkl",'wb') as pklFile:
		pickle.dump({'dataX':dataX,
									'dataY':dataY,
									'mass':mass, # for plotting
								 	'massBinned':massBinned},pklFile,-1)

if thingsToDo&2:
	if not(thingsToDo&1):
		with open(dataPath+"compressed.pkl",'rb') as pklFile:
			d = pickle.load(pklFile)
			dataX = d['dataX']
			dataY = d['dataY']
			massBinned = d['massBinned']
			mass = d['mass']
	print dataX[:10]
	nData = dataY.shape[0]
	nTrain = nData*7/8
	nTest = 1000
	nValidate = nData-nTrain-nTest
	# nValidate = nData*1/16
	learningRate = .001
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

	classifier = NN.NeuralNet(x,rng,[nVars,300,300,300,2])
	# trainer,loss = classifier.getTrainer(0,0,"NLL")
	trainer,loss = classifier.getRegularizedTrainer(0.8,"NLL+BGBinnedReg")
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
			if nSinceLastImprovement == 10:
				nSinceLastImprovement=0
				learningRate = learningRate*.8
				msgFile.write("\tLearningRate: %f\n"%(learningRate))
			idx = trainIndices[i*nPerBatch:(i+1)*nPerBatch]
			# print classifier.testFcn(massBinned[idx],dataY[idx],dataX[idx])
			# sys.exit(-1)
			trainer(dataX[idx],dataY[idx],learningRate,massBinned[idx])
			# trainer(dataX[idx],dataY[idx],learningRate)
			if not iteration%50:
				msgFile.write("Iteration: %i\n"%(iteration))
				# testLoss = loss(dataX[testIndices],dataY[testIndices])[0]
				testLoss = loss(dataX[testIndices],dataY[testIndices],massBinned[testIndices])[0]
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
			# if iteration > patience:
			if iteration > 1000:
				done=True
				break
			if learningRate < 0.0000001:
				done = True
				break
		if done:
			break
		epoch+=1

	classifier.initialize(bestParameters)
	print NN.evaluateZScore(classifier.probabilities(dataX[validateIndices]),dataY[validateIndices],mass[validateIndices],True)


