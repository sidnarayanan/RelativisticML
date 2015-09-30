import Classifiers.NeuralNet as NN
import cPickle as pickle
import numpy as np
import theano.tensor as T

rng = np.random.RandomState()
x = T.matrix('x')

classifier = NN.NeuralNet(x,rng,[5,5,2])

with open('bestParams.pkl','rb') as pklFile:
    bestParams = pickle.load(pklFile)
classifier.initialize(bestParams)
