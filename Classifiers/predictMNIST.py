#!/usr/bin/python

import cPickle as pickle
import numpy as np
import theano
import theano.tensor as T
import NeuralNet
import sys
from os import fsync

fTrain = open('../sampleData/mnist.pkl','rb')
# lossFile = open('loss_6h.log','w')
# msgFile = open('msg_6h.log','w')
# sys.stderr=msgFile
# sys.stdout=msgFile
lossFile = sys.stdout
msgFile = sys.stderr
realOutput = False
pklObj = pickle.load(fTrain)
rng = np.random.RandomState()
x = T.matrix('x')

classifier = NeuralNet.NeuralNet(x,rng,[784,1000,10])
# classifier = NeuralNet.NeuralNet(x,rng,[784,1000,1000,500,500,500,10])
dataTrain = pklObj[0]
dataTrainX = dataTrain[0]
dataTrainY = dataTrain[1]
dataTest = pklObj[1]
dataTestX = dataTest[0]
dataTestY = dataTest[1]

nTrain = dataTrainX.shape[0]
trainIndices = np.arange(nTrain)
msgFile.write("%d\n"%(nTrain))
lossFile.write("%f\n"%(classifier.errors(dataTestX,dataTestY)))
trainer = classifier.getTrainer(0,0,"NLL")
learningRate = 0.05
bestTestLoss = np.inf
epoch=0
iteration=0
nEpoch=1000
patienceBaseVal = 100000 # do at least this many iterations
patience = patienceBaseVal
patienceFactor = 1.5
significantImprovement = .995
done=False
nPerBatch=50
while (epoch<nEpoch):
	np.random.shuffle(trainIndices)
	if not epoch%20:
		learningRate = learningRate/2
	msgFile.write("Epoch: %i\n"%(epoch))
	for i in range(nTrain/nPerBatch):
		idx = trainIndices[i*nPerBatch:(i+1)*nPerBatch]
		trainer(dataTrainX[idx],dataTrainY[idx],learningRate)
		msgFile.write("Iteration: %i\n"%(iteration))
		testLoss = classifier.errors(dataTestX,dataTestY)
		sys.stdout.write("%f\n"%(testLoss))
		if not i%50:
			# msgFile.write("Iteration: %i\n"%(iteration))
			# testLoss = classifier.errors(dataTestX,dataTestY)
			# lossFile.write("%f\n"%(testLoss))
			if testLoss < bestTestLoss:
				msgFile.write("\tNewBestLoss: %f\n"%(testLoss))
				if testLoss/bestTestLoss < significantImprovement:
					patience = patienceBaseVal+iteration*patienceFactor
					msgFile.write("\tIncreasingPatience: %f\n"%(patience))
					bestTestLoss = testLoss
			print i
			# if realOutput:
			# 	msgFile.flush()
			# 	lossFile.flush()
			# 	fsync(msgFile.fileno())
			# 	fsync(lossFile.fileno())
		iteration+=1
		if iteration > patience:
			print iteration, patience
			done=True
			break
	if done:
		break
	epoch+=1

msgFile.close()
lossFile.close()
