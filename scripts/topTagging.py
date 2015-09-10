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

thingsToDo = 3 if len(sys.argv)==1 else int(sys.argv[1])

config.int_division = 'floatX'
def divide(a):
	return a[0]/a[1]
def bin(a,b,m):
	return min(int(a[0]/b),m)

lossFile = sys.stdout
msgFile = sys.stderr

rng = np.random.RandomState()
x = T.matrix('x')

listOfRawVars = ["massSoftDrop","QGTag","QjetVol","telescopingIso","groomedIso"]
listOfComputedVars = [(divide,['tau3','tau2']),
												(divide,['tau2','tau1']),
												(np.log,['chi'])]
nVars = len(listOfComputedVars) + len(listOfRawVars)

if thingsToDo&1:
	sigImporter = ROOTInterface.Import.TreeImporter('/home/sid/scratch/data/signal_AK8fj.root','jets')
	sigImporter.addVarList(listOfRawVars)
	for v in listOfComputedVars:
		sigImporter.addComputedVar(v)
	bgImporter = sigImporter.clone('/home/sid/scratch/data/qcd_AK8fj.root','jets')
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
	sigImporter.addComputedVar((massBin,['massSoftDrop']))
	bgImporter.addComputedVar((massBin,['massSoftDrop']))
	massBinned = np.vstack([sigImporter.loadTree(0,-1)[0],
										bgImporter.loadTree(0,-1)[0]])

	with open("/home/sid/scratch/data/topTagging_test.pkl",'wb') as pklFile:
		pickle.dump({'dataX':dataX,
									'dataY':dataY,
								 	'massBinned':massBinned},pklFile,-1)

if thingsToDo&2:
	if not(thingsToDo&1):
		with open("/home/sid/scratch/data/topTagging_test.pkl",'rb') as pklFile:
			d = pickle.load(pklFile)
			dataX = d['dataX']
			dataY = d['dataY']
			massBinned = d['massBinned']
	nData = dataY.shape[0]
	nTrain = nData*1/2
	nValidate = nData*1/4
	learningRate = .01
	nSinceLastImprovement = 0
	bestTestLoss = np.inf
	sigTestLoss = np.inf
	epoch=0
	iteration=0
	nEpoch=1000
	patienceBaseVal = 100000 # do at least this many iterations
	patience = patienceBaseVal
	patienceFactor = 1.5
	significantImprovement = .995
	done=False
	nPerBatch=200

	classifier = NN.NeuralNet(x,rng,[nVars,100,100,100,2])
	trainer = classifier.getRegularizedTrainer(0.1,"NLL+BGBinnedReg")
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
			trainer(dataX[idx],dataY[idx],learningRate,massBinned[idx,0])
			# trainer(dataX[idx],dataY[idx],learningRate)
			if not iteration%50:
				msgFile.write("Iteration: %i\n"%(iteration))
				testLoss = NN.evaluateZScore(classifier.probabilities(dataX[testIndices]),dataY[testIndices],None,False)
				# testLoss = classifier.errors(dataX[testIndices],dataY[testIndices])
				lossFile.write("%f\n"%(testLoss))
				if testLoss < bestTestLoss:
					nSinceLastImprovement=0
					msgFile.write("\tNewBestLoss: %f\n"%(testLoss))
					bestTestLoss = testLoss
					if testLoss/sigTestLoss < significantImprovement:
						patience = patienceBaseVal+iteration*patienceFactor
						msgFile.write("\tIncreasingPatience: %f\n"%(patience))
						sigTestLoss = testLoss
				else:
					nSinceLastImprovement+=1
				if iteration>400:
					done=True
					break
			iteration+=1
			if iteration > patience:
				print iteration, patience
				done=True
				break
			if learningRate < 0.0000001:
				done = True
				break
		if done:
			break
		epoch+=1

	print NN.evaluateZScore(classifier.probabilities(dataX[validateIndices]),dataY[validateIndices],None,False)


