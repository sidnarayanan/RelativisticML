import Classifiers.NeuralNet as NN
import cPickle as pickle
import numpy as np
import theano.tensor as T

rng = np.random.RandomState()
x = T.matrix('x')

nVars=7
hiddenSize = nVars*3
nHidden = 10
layers = [nVars]
for i in xrange(nHidden):
	layers.append(hiddenSize)
layers.append(2)
classifier = NN.NeuralNet(x,rng,layers)

with open('bestParams_0_470_CA15.pkl','rb') as pklFile:
    bestParams = pickle.load(pklFile)
classifier.initialize(bestParams)


dataPath = '/home/snarayan/cms/root/topTagging_CA15_weighted/noTopSize'

dataX = np.array([-30 , 0.0818020 , 0.0772602 , 0.2248658 , 0.9979527 , 0.9817413 , 0.6876764 , 0.8317915 ])
print dataX
'''
longSuffix = ('_ptGT%.1fANDptLT%.1fANDabsetaLT%.1f'%(0,470,2.4)).replace('.','p')
alphas = np.empty(nVars+1)
V = np.empty([nVars+1,nVars+1])
with open(dataPath+'/pca.txt') as pcaFile:
	for line in pcaFile:
		if line.find(longSuffix) >= 0:
			ll = line.split()
			if ll[0]=='alpha':
				alphas[int(ll[1])] = float(ll[-1])
			else:
				for i in xrange(nVars+1):
					# print i,ll[3+i]
					V[i,int(ll[1])] = float(ll[3+i])
truncV = V[:,1:]
dataX = np.dot(dataX,truncV)
print dataX
'''
with open(dataPath+"/compressedNoTopSize_0_470_2p4_small.pkl",'rb') as pklFile:
	d = pickle.load(pklFile)
	# dataX = d['dataX']
	# dataY = d['dataY']
	# massBinned = d['massBinned']
	# mass = d['mass']
	# nSig = d['nSig']
	# nBg = d['nBg']
	mu = d['mu']
	sigma = d['sigma']

dataX = np.array([(dataX-mu)/sigma])
print dataX
print classifier.probabilities(dataX)
