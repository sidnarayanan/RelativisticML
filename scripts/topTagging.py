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
def bin(a,b,m,M):
	# binwidth, min, max
	return int(max(min(a[0]-m,M-m),0)/b)

lossFile = sys.stdout
msgFile = sys.stderr

rng = np.random.RandomState()
x = T.matrix('x')

listOfRawVars = ["QGTag","maxSubjetBtag"]
listOfComputedVars = [(divide,['tau3','tau2']),
						(divide,['tau2','tau1'])]
# listOfRawVars += ["iota0","iota1","iota2","iota3","iota4"]
# listOfComputedVars = []
nVars = len(listOfComputedVars) + len(listOfRawVars)

dataPath = '/home/sid/scratch/data/topTagging_SDTopWidth25/'

if thingsToDo&1:
	nEntries = -1

	# kinematic cuts
	(massBinWidth,massMin,massMax) = (25,0,250)
	(ptBinWidth,ptMin,ptMax) = (50,800,2000)
	(etaBinWidth,etaMin,etaMax) = (.5,-1.5,1.5)

	# first import variables
	print "Loading variables"
	sigImporter = ROOTInterface.Import.TreeImporter(dataPath+'signal_AK8fj.root','jets')
	for v in listOfRawVars:
		sigImporter.addVar(v)
	for v in listOfComputedVars:
		sigImporter.addComputedVar(v)
	bgImporter = sigImporter.clone(dataPath+'qcd_AK8fj.root','jets')
	sigX,sigY = sigImporter.loadTree(1,nEntries)
	bgX,bgY = bgImporter.loadTree(0,nEntries)
	dataX = np.vstack([sigX,bgX])
	dataY = np.hstack([sigY,bgY])
	nSig = sigY.shape[0]
	nBg = bgY.shape[0]
	
	# next get  kinematics
	print "Loading kinematics"
	sigImporter.resetVars()
	bgImporter.resetVars()
	sigImporter.resetCounter()
	bgImporter.resetCounter()
	def massBin(a):
		return bin([a],massBinWidth,massMin,massMax)
	def ptBin(a):
		return bin([a],ptBinWidth,ptMin,ptMax)
	def etaBin(a):
		return bin([a],etaBinWidth,etaMin,etaMax)
	sigImporter.addVar('massSoftDrop')
	sigImporter.addVar('pt')
	sigImporter.addVar('eta')
	bgImporter = sigImporter.clone(dataPath+'qcd_AK8fj.root','jets')
	sigVars = sigImporter.loadTree(0,nEntries)[0]
	bgVars = bgImporter.loadTree(0,nEntries)[0]

	# store kinematics and apply cuts
	print "Applying cuts"
	mass = np.hstack([sigVars[:,0],bgVars[:,0]])
	goodEntries = np.array([(massMin<m<massMax) for m in mass])
	mass = mass[goodEntries]
	massBinned = np.array([massBin(m) for m in mass])
	massMask = np.array([(140<m<210) for m in mass])
	pt = np.hstack([sigVars[:,1],bgVars[:,1]])[goodEntries]
	ptBinned = np.array([ptBin(p) for p in pt])
	eta = np.hstack([sigVars[:,2],bgVars[:,2]])[goodEntries]
	etaBinned = np.array([etaBin(e) for e in eta])
	print mass[:20]
	print massBinned[:20]

	# propogate cuts
	dataX = dataX[goodEntries]
	dataY = dataY[goodEntries]

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

	# weight events based on mSD, signal, etc:
	print "Loading weights"
	sigImporter.resetCounter()
	sigImporter.resetVars()
	bgImporter.resetCounter()
	bgImporter.resetVars()
	sigImporter.addVar('massWeight')
	bgImporter.addVar('massWeight')
	weight = np.hstack([sigImporter.loadTree(0,nEntries)[0][:,0]*nBg/nSig,
						bgImporter.loadTree(0,nEntries)[0][:,0]])[goodEntries]


	with open(dataPath+"compressed_nomass.pkl",'wb') as pklFile:
		pickle.dump({'nSig':nSig,  'nBg':nBg, 
									'dataX':dataX,
									'dataY':dataY,
									'weight':weight,
									'mass':mass, 'massBinned':massBinned,
									'pt':pt, 'ptBinned':ptBinned,
									'eta':eta, 'etaBinned':etaBinned},pklFile,-1)

if thingsToDo&2:
	if not(thingsToDo&1):
		with open(dataPath+"compressed_nomass.pkl",'rb') as pklFile:
			d = pickle.load(pklFile)
			dataX = d['dataX']
			dataY = d['dataY']
			weight = d['weight']
			massBinned = d['massBinned']
			mass = d['mass']
			nSig = d['nSig']
			nBg = d['nBg']
	weight = weight*10000.
	# massWindow = np.array([1 if (140<m<210) else 0 for m in mass])
	massWindow = np.ones(dataY.shape[0])
	print dataX[:10]
	nData = dataY.shape[0]
	nTrain = nData*1/2-5000
	nTest = 10000.
	nValidate = nData-nTrain-nTest
	print nTrain
	# nValidate = nData*1/16
	learningRate = .001
	nSinceLastImprovement = 0
	bestTestLoss = np.inf
	sigTestLoss = np.inf
	epoch=0
	iteration=0
	nEpoch=5
	patienceBaseVal = 10000 # do at least this many iterations
	patience = patienceBaseVal
	patienceFactor = 1.1
	significantImprovement = .995
	done=False
	nPerBatch=2000

	hiddenSize = nVars*5
	nHidden = 5
	layers = [nVars]
	for i in xrange(nHidden):
		layers.append(hiddenSize)
	layers.append(2)
	classifier = NN.NeuralNet(x,rng,layers)
	# classifier.setSignalWeight(float(nBg)/nSig)

	# trainer,loss = classifier.getTrainer(0,0,"NLL")
	regStrength = 0
	# trainer,loss,reg = classifier.getRegularizedTrainer(regStrength,"WeightedNLL")
	trainer,loss,reg = classifier.getWindowedTrainer(regStrength,"WWNLL")
	# trainer,loss,reg = classifier.getRegularizedTrainer(regStrength,"WeightedNLL+BGBinnedYield")
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
			idx = trainIndices[i*nPerBatch:(i+1)*nPerBatch]
			# print classifier.testFcn(massBinned[idx],dataY[idx],dataX[idx])
			# sys.exit(-1)
			trainer(dataX[idx],dataY[idx],learningRate,massBinned[idx],massWindow[idx],weight[idx])
			# trainer(dataX[idx],dataY[idx],learningRate,massBinned[idx])
			# trainer(dataX[idx],dataY[idx],learningRate)
			if not iteration%50:
				msgFile.write("Iteration: %i\n"%(iteration))
				# testLoss = loss(dataX[testIndices],dataY[testIndices])[0]
				testLoss = loss(dataX[testIndices],dataY[testIndices],massBinned[testIndices],massWindow[testIndices],weight[testIndices])[0]
				# testLoss = loss(dataX[testIndices],dataY[testIndices],massBinned[testIndices])[0]
				if reg:
					testReg = reg(dataX[testIndices],dataY[testIndices],massBinned[testIndices])[0]
				# testLoss = NN.evaluateZScore(classifier.probabilities(dataX[testIndices]),dataY[testIndices],None,False)
				# testLoss = classifier.errors(dataX[testIndices],dataY[testIndices])
				if reg:
					lossFile.write("%f %f\n"%(testLoss,regStrength*testReg))
					sh =  classifier.evalSelectedHist(dataX[testIndices],dataY[testIndices],massBinned[testIndices])[0]
					shVal = (sh-np.mean(sh))
					print np.mean(sh),np.sum(sh),np.std(sh), np.mean(shVal*shVal)
				else:
					lossFile.write("%f\n"%(testLoss))
				if testLoss<1.5:
					learningRate = 0.001
					msgFile.write("\tLearningRate: %f\n"%(learningRate))
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
			if learningRate < 0.00001:
				done = True
				break
		if done:
			break
		epoch+=1

	classifier.initialize(bestParameters)
	print NN.evaluateZScore(classifier.probabilities(dataX[validateIndices]),dataY[validateIndices],mass[validateIndices],True)


	with open("bestParams.pkl",'wb') as pklFile:
		pickle.dump(bestParameters,pklFile,-1)

	with open("topTagger.icc","w") as fOut:
		exporter = ROOTInterface.Export.NetworkExporter(classifier)
		exporter.setFile(fOut)
		exporter.export('topANN')