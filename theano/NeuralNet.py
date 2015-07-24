#!/usr/bin/python

import theano
import theano.tensor as T
import numpy as np
import Logistic

theano.config.int_division = 'floatX'

class HiddenLayer(object):
	def __init__(self,input,rng,nIn,nOut,W=None,b=None,sigmoid=T.tanh):
		if not W:
			# initialize W and allocate it
			W0 = np.asarray(rng.uniform(
								low=-np.sqrt(6./(nIn+nOut)),
								high=np.sqrt(6./(nIn+nOut)),
								size=(nIn,nOut)
							), dtype=theano.config.floatX)
			if not sigmoid==T.tanh:
				W0 = 4*W0
			W = theano.shared(value=W0,name='W',borrow=True)
		if not b:
			# initialize b and allocate it
			b0 = np.zeros((nOut,),dtype=theano.config.floatX)
			b = theano.shared(value=b0,name='b',borrow=True)
		self.W = W
		self.b = b
		if sigmoid:
			self.output = sigmoid(T.dot(input,self.W) + self.b)
		else:
			self.output = T.dot(input,self.W) + self.b

class NeuralNet(object):
	def __init__(self,input,rng,layerSize):
		'''
			network architecture:
			layerSize[0] -> ... -> layerSize[N-1]
		'''
		N = len(layerSize)
		self.input=input
		self.hiddenLayers=[HiddenLayer(
				input=input,
				rng=rng,
				nIn=layerSize[0],
				nOut=layerSize[1]
			)]
		for i in range(1,N-2):
			self.hiddenLayers.append(
					HiddenLayer(
							input=self.hiddenLayers[i-1].output,
							rng=rng,
							nIn=layerSize[i],
							nOut=layerSize[i+1]
						)
				)
		self.outLayer = Logistic.Logistic(nIn=layerSize[N-2],
											nOut=layerSize[N-1],
											x=self.hiddenLayers[-1].output)
		self.L1 = abs(self.outLayer.W).sum()
		L2_sqr = (self.outLayer.W**2).sum()
		for hl in self.hiddenLayers:
			self.L1+=abs(hl.W).sum()
			L2_sqr+=(hl.W**2).sum()
		self.L2 = T.sqrt(L2_sqr)
		self.NLL = self.outLayer.NLL
		self.MSE = self.outLayer.MSE
		self.BoverS2 = self.outLayer.BoverS2
		self.BGReg = self.outLayer.BGReg
		self.BGBinnedReg = self.outLayer.BGBinnedReg
		y = T.ivector('y')
		self.errors = theano.function(
				inputs=[self.input,y],
				outputs=T.mean(T.neq(self.outLayer.yHat,y)),
				givens={},
        		allow_input_downcast=True
			)
		self.evaluate = theano.function(
				inputs=[self.input],
				outputs=self.outLayer.yHat
			)
		self.probabilities = theano.function(
				inputs = [self.input],
				outputs=self.outLayer.P
			)
		self.theta = []
		for hl in self.hiddenLayers:
			self.theta += [hl.W,hl.b]
		self.theta += [self.outLayer.W,self.outLayer.b]
	def testFcn(self,massBinned,trainY,trainX):
		y = T.dvector('y')
		varBinned = T.ivector('var')
		baseHist = T.bincount(varBinned,1-y)+0.01
		selectedHist = T.bincount(varBinned,(1-y)*self.outLayer.P[T.arange(y.shape[0]),1])+0.01
		print baseHist.eval({y:trainY, varBinned:massBinned}), selectedHist.eval({y:trainY, varBinned:massBinned, self.input:trainX})
		rTensor = T.std(selectedHist/baseHist)
		return (rTensor).eval({y:trainY, varBinned:massBinned, self.input:trainX})
	def evalNLL(self,testX,testY):
		return T.log(self.outLayer.P).eval({self.input:testX})
	def getRegularizedTrainer(self,bgRegStrength=0, errorType="NLL+BGBinnedReg"):
		trainY = T.ivector('y')
		var = T.ivector('var')
		alpha = T.dscalar('a')
		if errorType=="NLL+BGReg":
			loss = self.NLL(trainY) + bgRegStrength * self.BGReg(trainY)
		elif errorType=="NLL+BGBinnedReg":
			loss = self.NLL(trainY) + bgRegStrength * self.BGBinnedReg(trainY,var)
		else:
			return None
		dtheta = [T.grad(cost=loss,wrt=x) for x in self.theta]
		updates = [(self.theta[i],
					self.theta[i]-alpha*dtheta[i])
					for i in range(len(dtheta))]
		trainer = theano.function(
				inputs = [self.input,trainY,alpha,var],
				outputs = [loss],
				updates = updates,
				givens = { },
				allow_input_downcast=True,
				on_unused_input='warn'
			)
		return trainer
	def getTrainer(self,L1Reg,L2Reg,errorType="NLL"):
		trainY = T.ivector('y')
		alpha = T.dscalar('a')
		if errorType=="NLL":
			loss = self.NLL(trainY) + L1Reg*self.L1 + L2Reg*self.L2
		elif errorType=="MSE":
			loss = self.MSE(trainY)
		elif errorType=="BoverS2":
			loss = self.BoverS2(trainY)
		else:
			return None
		dtheta = [T.grad(cost=loss,wrt=x) for x in self.theta]
		updates = [(self.theta[i],
					self.theta[i]-alpha*dtheta[i])
					for i in range(len(dtheta))]
		trainer = theano.function(
				inputs = [self.input,trainY,alpha],
				outputs = [loss],
				updates = updates,
				givens = { },
				allow_input_downcast=True
			)
		return trainer
