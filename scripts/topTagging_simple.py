#!/usr/bin/python

import cPickle as pickle
import numpy as np
from theano import config
import theano.tensor as T
import Classifiers.NeuralNet as NN
# import ROOTInterface.Import
import ROOTInterface.Export
import sys
import ROOT as root # turned off to run on t3
from os import fsync


# thingsToDo = 0 if len(sys.argv)==1 else int(sys.argv[1])

if not(len(sys.argv)==5):
	print 'usage: %s ptlow pthigh absetahigh algo'%(sys.argv[0])
	sys.exit(1)
else:
	ptlow = int(sys.argv[1])
	pthigh = int(sys.argv[2])
	etahigh = float(sys.argv[3])
	jetAlgo = sys.argv[4]


print '%f < pT < %f && |eta| < %f, %s'%(ptlow,pthigh,etahigh,jetAlgo)



config.int_division = 'floatX'
def divide(a):
	return a[0]/a[1]
def bin(a,b,m):
	return min(int(a[0]/b),m)

lossFile = sys.stdout
msgFile = sys.stderr

rng = np.random.RandomState()
x = T.matrix('x')

# listOfRawVars = ["logchi","QGTag","QjetVol","groomedIso"]
# listOfComputedVars = [(divide,['tau3','tau2'])]
# nVars = len(listOfComputedVars) + len(listOfRawVars)

dataPath = '/home/snarayan/cms/root/topTagging_%s/'%(jetAlgo)

# dataPath = '/home/sid/scratch/data/topTagging_%s/'%(jetAlgo)

suffix = '%i_%i_%.1f'%(ptlow,pthigh,etahigh)
suffix = suffix.replace('.','p')
# dataPath = '/home/snarayan/cms/root/topTagging_CA15/'
with open(dataPath+"compressedWeighted_%s.pkl"%(suffix),'rb') as pklFile:
	print 'loading data!'
	d = pickle.load(pklFile)
	dataX = d['dataX']
	dataY = d['dataY']
	# massBinned = d['massBinned']
	kinematics = d['kinematics']
	nSig = d['nSig']
	nBg = d['nBg']
	vars = d['vars']
	mu = d['mu']
	sigma = d['sigma']
	weight = d['weights']

nVars = len(vars)# -1 # -1 if using PCA
print vars

#apply cuts
# mass = kinematics[:,0]
# pt = kinematics[:,1]
# eta = kinematics[:,2]
# cut = np.logical_and(np.logical_and(pt<pthigh,pt>ptlow),np.abs(eta)<etahigh)
# dataX = dataX[cut]
# dataY = dataY[cut]
# kinematics = kinematics[cut]

print dataX[:10]
# weight = weight*10000.
nData = dataY.shape[0]
nSig = int(np.sum(dataY))
nBg = nData-nSig
print nSig,nBg

scale = np.hstack([0.1*np.ones(nSig),np.ones(nBg)])
weight = scale*weight

nValidate = 3000
nTest = 10000
nTrain = nData-nTest-nValidate
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

# dimensions
hiddenSize = nVars*3
nHidden = 10
layers = [nVars]
for i in xrange(nHidden):
	layers.append(hiddenSize)
layers.append(2)
classifier = NN.NeuralNet(x,rng,layers)
classifier.vars = vars
# classifier.setSignalWeight(float(nBg)/nSig)
trainer,loss = classifier.getTrainer(0,0,"WeightedNLL")
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
			learningRate = learningRate*.1
			msgFile.write("\tLearningRate: %f\n"%(learningRate))
			classifier.initialize(bestParameters) # go back to the best point
		idx = trainIndices[i*nPerBatch:(i+1)*nPerBatch]
		trainer(dataX[idx],dataY[idx],learningRate,weight[idx])
		if not iteration%50:
			msgFile.write("Iteration: %i\n"%(iteration))
			testLoss = loss(dataX[testIndices],dataY[testIndices],weight[testIndices])[0]
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
		if learningRate <= 0.00001:
			done = True
			break
	if done:
		break
	epoch+=1

classifier.initialize(bestParameters)
print NN.evaluateZScore(classifier.probabilities(dataX[validateIndices]),dataY[validateIndices],kinematics[validateIndices,0],True)

fileName = "%i_%i_%s"%(ptlow,pthigh,jetAlgo)
# fileName = fileName.replace('.','p')

with open("bestParams_%s.pkl"%(fileName),'wb') as pklFile:
	pickle.dump(bestParameters,pklFile,-1)

with open("topTagger_%s.icc"%(fileName),"w") as fOut:
	exporter = ROOTInterface.Export.NetworkExporter(classifier)
	exporter.setFile(fOut)
	exporter.export('topANN_%s'%(fileName),mu,sigma)
