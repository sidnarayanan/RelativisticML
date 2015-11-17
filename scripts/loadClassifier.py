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


dataPath = '/home/snarayan/cms/root/topTagging_CA15/'

dataX = np.array([-30.,0.5546839,0.0118666,0.0032285,0.9616043,0.9989137,0.9970202,0.3705301])
dataX = np.array([[-16.27255 , 0.0111608 , 0.0303662 , 0.0573616 , 0.0348162 , 0.9767509 , 0.0108862 , 0.7515959],
[-19.12756 , 3.575e-08 , 0.0411837 , 0.2201423 ,         0 , 0.9931696 , 0.0108862 , 0.6967631],
[      -30 , 0.0217773 , 0.0184720 , 0.0149450 , 0.2820092 , 0.5808168 , 1.538e-08 , 0.5940328],
[      -30 , 0.3418268 , 0.0935915 , 0.0886995 , 0.2744155 ,         1 , 1.538e-08 , 0.6767584],
[      -30 , 0.0112696 , 0.0524010 , 0.1188580 , 0.0817418 , 0.5310187 , 1.538e-08 , 0.8750675],
[-13.81437 , 0.0033988 , 0.0974235 , 0.1611052 , 0.1260528 , 0.0391818 , 0.8926063 , 0.6661009]])

print dataX.shape
longSuffix = ('_ptGT%.1fANDptLT%.1fANDabsetaLT%.1f'%(0,470,2.4)).replace('.','p')
alphas = np.empty(nVars+1)
V = np.empty([nVars+1,nVars+1])
with open(dataPath+"/compressedWindowPtWeighted_0_470_2p4_small.pkl",'rb') as pklFile:
  d = pickle.load(pklFile)
  # dataX = d['dataX']
  # dataY = d['dataY']
  # massBinned = d['massBinned']
  # mass = d['mass']
  # nSig = d['nSig']
  # nBg = d['nBg']
  mu = d['mu']
  sigma = d['sigma']

dataX = (dataX-mu)/sigma
print mu
print sigma
print dataX
print classifier.probabilities(dataX)
