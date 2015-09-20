#!/usr/bin/python

import theano
import theano.tensor as T
import numpy as np
import Logistic
import ROOT as root

theano.config.int_division = 'floatX'

def evaluateZScore(probabilities,truth,prunedMass,makePlots=False):
	aSig = probabilities[truth==1][:,1]
	aBg = probabilities[truth==0][:,1]
	aSig.sort()
	cutVal = aSig[int(aSig.shape[0]/2)]
	bgPassed = float(aBg[aBg>cutVal].shape[0])
	zScore = bgPassed/aBg.shape[0]
	if makePlots:
		fout = root.TFile("outputHists.root","RECREATE")
		hSig = root.TH1F("hSig","hSig",100,0.,1.)
		hBg = root.TH1F("hBg","hBg",100,0.,1.)
		hMassSig = root.TH1F("hMassSig","hMassSig",100,0,300)
		hMassBg = root.TH1F("hMassBg","hMassBg",100,0,300)
		for i in xrange(truth.shape[0]):
			if truth[i]==1:
				hSig.Fill(probabilities[i,1])
				if probabilities[i,1] > cutVal:
					hMassSig.Fill(prunedMass[i])
			else:
				hBg.Fill(probabilities[i,1])
				if probabilities[i,1] > cutVal:
					hMassBg.Fill(prunedMass[i])
		hBg.SetLineColor(2)
		fout.WriteTObject(hSig,"hSig")
		fout.WriteTObject(hBg,"hBg")
		c1 = root.TCanvas()
		hMassSig.SetNormFactor()
		hMassSig.Draw("")
		hMassBg.SetNormFactor()
		hMassBg.SetLineColor(2)
		hMassBg.Draw("same")
		c1.SaveAs("mass.png")
		fout.Write()
		fout.Close()
	return zScore

class HiddenLayer(object):
	def __init__(self,input,rng,nIn,nOut,W=None,b=None,sigmoid=T.tanh):
		if not W:
			# initialize W and allocate it
			W0 = np.asarray(rng.uniform(low=-np.sqrt(6./(nIn+nOut)),
										high=np.sqrt(6./(nIn+nOut)),
										size=(nIn,nOut)), 
							dtype=theano.config.floatX)
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
	def __getstate__(self):
		return {'W':self.W.get_value(),
				'b':self.b.get_value()}
	def __setstate__(self,d):
		self.W.set_value(d['W'])
		self.b.set_value(d['b'])

class NeuralNet(object):
	def __init__(self,input,rng,layerSize):
		if len(layerSize)<3:
			print "Warning: cannot create NeuralNet without at least one hidden layer (use Classifiers.Logistic.Logistic instead)"
		'''
			network structure:
			layerSize[0] -> ... -> layerSize[N-1]
		'''
		self.nIn = layerSize[0]
		self.nOut = layerSize[-1]
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
	def getParameters(self):
		# easy way of exporting network parameters
		params=[]
		for l in self.hiddenLayers+[self.outLayer]:
			p={}
			p['W'] = l.W.get_value()
			p['b'] = l.b.get_value()
			params.append(p)
		return params
	def initialize(self,parameterList):
		if not len(parameterList)==(len(self.hiddenLayers)+1):
			print "Could not initialize"
			return
		for l,p in zip(self.hiddenLayers+[self.outLayer],parameterList):
			l.W.set_value(p['W'])
			l.b.set_value(p['b'])
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
		evalLoss = theano.function(
				inputs = [self.input,trainY,var],
				outputs = [loss],
				updates = { },
				givens = { },
				allow_input_downcast=True,
				on_unused_input='warn'
			)
		return trainer,evalLoss
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
		evalLoss = theano.function(
				inputs = [self.input,trainY],
				outputs = [loss],
				updates = { },
				givens = { },
				allow_input_downcast=True,
				on_unused_input='warn'
			)
		return trainer,evalLoss