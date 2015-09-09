#!/usr/bin/python

import theano
import theano.tensor as T
import numpy as np
import Logistic


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

class MLP(object):
	def __init__(self,input,rng,nIn,nHidden,nOut):
		'''
			network architecture:
			nIn -> nHidden -> nOut
		'''
		self.input=input
		self.hiddenLayer = HiddenLayer(input=input,
										rng=rng,
										nIn=nIn,
										nOut=nHidden,
										sigmoid=T.tanh)
		self.outLayer = Logistic.Logistic(nIn=nHidden, 
											nOut=nOut, 
											x=self.hiddenLayer.output)
		self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.outLayer.W).sum()
		self.L2 = T.sqrt((self.hiddenLayer.W**2).sum() + (self.outLayer.W**2).sum())
		self.NLL = self.outLayer.NLL
		y = T.ivector('y')
		self.errors = theano.function(
				inputs=[self.input,y], 
				outputs=T.mean(T.neq(self.outLayer.yHat,y)),
				givens={

				},
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
		self.theta = [self.hiddenLayer.W,self.hiddenLayer.b,
						self.outLayer.W,self.outLayer.b]
	def evalNLL(self,testX,testY):
		return T.log(self.outLayer.P).eval({self.input:testX})
	def getTrainer(self,L1Reg,L2Reg):
		trainY = T.ivector('y')
		alpha = T.dscalar('a')
		loss = self.NLL(trainY) + L1Reg*self.L1 + L2Reg*self.L2
		dtheta = [T.grad(cost=loss,wrt=x) for x in self.theta]
		updates = [(self.theta[i],
					self.theta[i]-alpha*dtheta[i])
					for i in range(len(dtheta))]
		trainer = theano.function(
				inputs = [self.input,trainY,alpha],
				outputs = [loss]+self.theta,
				updates = updates,
				givens = { },
				allow_input_downcast=True
			)
		return trainer
