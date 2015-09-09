import ROOTInterface.Export as Export
import Classifiers.NeuralNet as NN
import numpy as np
import theano.tensor as T
from sys import stdout


rng = np.random.RandomState()
x = T.matrix('x')
classifier = NN.NeuralNet(x,rng,[2,2,2])

exporter = Export.NetworkExporter(classifier)
exporter.setFile(stdout)
exporter.export('test')
