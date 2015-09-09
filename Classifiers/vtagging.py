#!/usr/bin/python

import cPickle as pickle
import numpy as np
import theano
import theano.tensor as T
import NeuralNet
import sys
import ROOT as root # turned off to run on t3
from os import fsync
from math import isnan

theano.config.int_division = 'floatX'

def evaluateZScore(probabilities,truth,prunedMass,plotMass=False):
	hSig = root.TH1F("hSig","hSig",100,0.,1.)
	hBg = root.TH1F("hBg","hBg",100,0.,1.)
	for i in xrange(truth.shape[0]):
		if truth[i]==1:
			hSig.Fill(probabilities[i,1])
		else:
			hBg.Fill(probabilities[i,1])
	nSig = hSig.Integral()
	nBg = hBg.Integral()
	nerrBg = 1
	done = False
	cutVal = 0
	for margin in [(.49,.51), (.45,.55), (.4,.6)]:
		for cut in xrange(100):
			nerrSig = hSig.Integral(1,cut)/nSig
			if margin[0] <nerrSig < margin[1]:
				nerrBg = hBg.Integral(cut,100)/nBg
				cutVal = cut
				done = True
				break
		if done:
			break
	if plotMass:
		fout = root.TFile("outputHists.root","RECREATE")
		fout.cd()
		hMassSig = root.TH1F("hMassSig","hMassSig",100,0,200)
		hMassBg = root.TH1F("hMassBg","hMassBg",100,0,200)
		for i in xrange(truth.shape[0]):
			if probabilities[i,1] > cutVal*0.01:
				if truth[i]==1:
					hMassSig.Fill(prunedMass[i])
				else:
					hMassBg.Fill(prunedMass[i])
		c1 = root.TCanvas()
		hMassSig.SetNormFactor()
		hMassSig.Draw("")
		hMassBg.SetNormFactor()
		hMassBg.SetLineColor(2)
		hMassBg.Draw("same")
		c1.SaveAs("mass.png")
		fout.Write()
		fout.Close()
	return nerrBg

lossFile = open('loss_3h.log','w')
msgFile = open('msg_3h.log','w')
realOutput = True
sys.stderr=msgFile
sys.stdout=msgFile
#
# realOutput = False
# lossFile = sys.stdout
# msgFile = sys.stderr

rng = np.random.RandomState()
x = T.matrix('x')

useCompressedData = True

# listOfRawVars = ["fjet1QGtagSub1","fjet1QGtagSub2","fjet1QGtag","fjet1PullAngle",
# 				"fjet1Pull","fjet1MassTrimmed","fjet1MassPruned","fjet1MassSDbm1",
# 				"fjet1MassSDb2","fjet1MassSDb0","fjet1QJetVol","fjet1C2b2","fjet1C2b1",
# 				"fjet1C2b0p5","fjet1C2b0p2","fjet1C2b0","fjet1Tau2","fjet1Tau1"]

listOfRawVars = ["fjet1QGtagSub1","fjet1QGtagSub2","fjet1QGtag","fjet1PullAngle",
				"fjet1Pull","fjet1QJetVol","fjet1C2b2","fjet1C2b1",
				"fjet1C2b0p5","fjet1C2b0p2","fjet1C2b0","fjet1Tau2","fjet1Tau1"]

nRawVars = len(listOfRawVars)

if useCompressedData:
	with open('/home/sid/scratch/data/vtagging_nomass.pkl','rb') as pklFile:
	# with open('/local/snarayan/local_scratch/vtagging.pkl','rb') as pklFile:
		pklObj = pickle.load(pklFile)
		dataX = pklObj['dataX']
		dataY = pklObj['dataY']
		mass = pklObj['mass']
		nData = dataX.shape[0]
else:
	sigFile = root.TFile('/tmp/ml/project/signal_word.root')
	bgFile = root.TFile('/tmp/ml/project/background_word.root')
	sigTree = sigFile.Get("DMSTree")
	bgTree = bgFile.Get("DMSTree")
	sigLeaves = sigTree.GetListOfLeaves()
	bgLeaves = bgTree.GetListOfLeaves()
	sigDict={}
	bgDict={}
	for i in range(sigLeaves.GetEntries()) :
	  leaf = sigLeaves.At(i)
	  sigDict[leaf.GetName()] = leaf
	for i in range(bgLeaves.GetEntries()) :
	  leaf = bgLeaves.At(i)
	  bgDict[leaf.GetName()] = leaf
	dataX = np.empty([sigTree.GetEntries()+bgTree.GetEntries(),nRawVars+3]) # two extra computed
	mass = np.empty([sigTree.GetEntries()+bgTree.GetEntries()])
	dataY = []
	nData = 0
	for n in range(sigTree.GetEntries()):
		sigTree.GetEntry(n)
		fjet1PartonId = sigDict["fjet1PartonId"].GetValue()
		if abs(fjet1PartonId)!=24:
			continue
		goodEvent=True
		m=0
		for name in listOfRawVars:
			dataX[nData,m]=sigDict[name].GetValue()
			if isnan(dataX[nData,m]):
				print "WARNING, event ",n,name," is nan in signal, skipping!"
				goodEvent=False
				break
			m+=1
		mass[nData] = sigDict["fjet1MassPruned"].GetValue()
		if isnan(mass[nData]):
			print "WARNING, event ",n," fjet1MassPruned is nan in signal, skipping!"
			goodEvent = False
		if not goodEvent:
			continue
		else:
			dataX[nData,nRawVars] = 2*dataX[nData,1] + dataX[nData,0] # 2*fjet1QGtagSub2+fjet1QGtagSub1
			dataX[nData,nRawVars+1] = float(dataX[nData,nRawVars-2])/dataX[nData,nRawVars-1] # tau2/tau1
			dataX[nData,nRawVars+2] = 1.0
			dataY.append(1)
			nData+=1
	for n in range(bgTree.GetEntries()):
		bgTree.GetEntry(n)
		fjet1PartonId = bgDict["fjet1PartonId"].GetValue()
		goodEvent=True
		m=0
		for name in listOfRawVars:
			dataX[nData,m]=bgDict[name].GetValue()
			if isnan(dataX[nData,m]):
				print "WARNING, event ",n,name," is nan in signal, skipping!"
				goodEvent=False
				break
			m+=1
		mass[nData] = bgDict["fjet1MassPruned"].GetValue()
		if isnan(mass[nData]):
			print "WARNING, event ",n," fjet1MassPruned is nan in background, skipping!"
			goodEvent = False
		if not goodEvent:
			continue
		else:
			dataX[nData,nRawVars] = 2*dataX[nData,1] + dataX[nData,0] # 2*fjet1QGtagSub2+fjet1QGtagSub1
			dataX[nData,nRawVars+1] = float(dataX[nData,nRawVars-2])/dataX[nData,nRawVars-1] # tau2/tau1
			dataX[nData,nRawVars+2] = 1.0
			dataY.append(0)
			nData+=1
	dataX = dataX[:nData]
	dataY = np.array(dataY)
	mass = mass[:nData]
	dataMean = dataX.mean(0)
	dataStd = dataX.std(0)
	for i in xrange(dataStd.shape[0]):
		# for constant rows, do not offset
		if not dataStd[i]:
			dataStd[i] = 1
			dataMean[i] = 0
	dataX = (dataX - dataMean)/dataStd
	with open("/home/sid/scratch/data/vtagging_nomass.pkl",'wb') as pklFile:
		pickle.dump({'dataX': dataX, 'dataY':dataY, 'mass':mass},pklFile,-1)
	sys.exit(-1)

print "Accepted %i events!"%(nData)

massBinned = np.empty([mass.shape[0]],dtype=np.int32)
for i in xrange(mass.shape[0]):
	# massBinned[i] = i%2
	massBinned[i] = min(int(mass[i]/10),15)
# print np.max(mass)

classifier = NeuralNet.NeuralNet(x,rng,[nRawVars+3,100,100,100,100,100,100,2])

nTrain = nData*3/4
# trainer = classifier.getTrainer(0,0,"NLL")
trainer = classifier.getRegularizedTrainer(0.5,"NLL+BGBinnedReg")
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
print "Done with initialization!"

dataIndices = np.arange(nData)
np.random.shuffle(dataIndices) # mix up signal and background
trainIndices = dataIndices[:nTrain]
testIndices = dataIndices[nTrain:]
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
		trainer(dataX[idx],dataY[idx],learningRate,massBinned[idx])
		# trainer(dataX[idx],dataY[idx],learningRate)
		if not iteration%50:
			msgFile.write("Iteration: %i\n"%(iteration))
			testLoss = evaluateZScore(classifier.probabilities(dataX[testIndices]),dataY[testIndices],mass[testIndices],False)
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
		if realOutput:
			msgFile.flush()
			lossFile.flush()
			fsync(msgFile.fileno())
			fsync(lossFile.fileno())
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

testLoss = evaluateZScore(classifier.probabilities(dataX[testIndices]),dataY[testIndices],mass[testIndices],True)


msgFile.close()
lossFile.close()
