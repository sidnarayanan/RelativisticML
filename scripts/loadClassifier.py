import Classifiers.NeuralNet as NN
import cPickle as pickle
import numpy as np
import theano.tensor as T

rng = np.random.RandomState()
x = T.matrix('x')

nVars=4
hiddenSize = nVars*5
nHidden = 5
layers = [nVars]
for i in xrange(nHidden):
	layers.append(hiddenSize)
layers.append(2)
classifier = NN.NeuralNet(x,rng,layers)

with open('bestParams.pkl','rb') as pklFile:
    bestParams = pickle.load(pklFile)
classifier.initialize(bestParams)


dataPath = '/home/sid/scratch/data/topTagging_SDTopWidth25/'

with open(dataPath+"compressed_nomass.pkl",'rb') as pklFile:
	d = pickle.load(pklFile)
	dataX = d['dataX']
	dataY = d['dataY']
	massBinned = d['massBinned']
	mass = d['mass']
	nSig = d['nSig']
	nBg = d['nBg']


print NN.evaluateZScore(classifier.probabilities(dataX),dataY,mass,True)
