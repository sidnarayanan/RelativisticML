#!/usr/bin/env python

import numpy as np

def boolToInt(b):
    if b:
        return 1
    else:
        return -1

class DecisionStump(object):
    """docstring for DecisionStump"""
    def __init__(self, nDim):
        self.nDim = nDim
        self.direction=True
        self.val=0
    def evaluate(self,x,val,direction):
        # this assumes y=0,1
        xhat=x[self.nDim]
        if xhat > val:
            return boolToInt(direction)
        else:
            return boolToInt(not direction)
    def evalDefault(self,x):
        return self.evaluate(x,self.val,self.direction)
    def trainGreedy(self,X,Y,W=None):
        xhat = X[:,self.nDim]
        minx = min(xhat)
        maxx = max(xhat)
        bestval=self.val
        bestdir=self.direction
        besterr=1
        N = len(Y)
        for i in range(100):
            testval= minx + (maxx-minx)*0.01
            for testdir in [True,False]:
                err=0.
                if W==None:
                    for j in range(N):
                        err+=abs(self.evaluate(X[j],testval,testdir) - Y[j])/float(N)
                else:
                    self.val=testval
                    self.direction=testdir
                    err = self.weightedError(X,Y,W)
                if err < besterr:
                    bestval=testval
                    besterr=err
                    bestdir=testdir
        self.val=bestval
        self.direction=bestdir
    def weightedError(self,X,Y,W):
        Wtot = sum(W)
        werr=0.
        for x,y,w in zip(X,Y,W):
            if not self.evalDefault(x)==y:
                werr+=w
        werr = werr/Wtot
        return werr
    def __str__(self):
        return "%i %.3f %i"%(self.nDim,self.val,boolToInt(self.direction))

# class DecisionNode(object):
#     def __init__(self,stump,nodePos,nodeNeg):
#         self.stump=stump
#         self.nodePos=nodePos
#         self.nodeNeg=nodeNeg

# class DecisionTree(object):
#     def __init__(self,dim,depth):
#         self.dim=dim
#         self.depth=depth
#         self.nodes=set([])
#         self.rootNode=None
#     def trainGreedy(X,Y,W):
#         newStump = DecisionStump(np.random.randint(0,self.dim))
#         self.rootNode = DecisionNode(newStump,None,None)
#         self.rootNode.trainGreedy(X,Y,W)

class Forest(object):
    def __init__(self,dim,nClassifiers,X,W=None,forestType="stump"):
        self.nClassifiers=nClassifiers
        self.dim=dim
        self.classifiers=[]
        self.classifierWeights=[]
        self.N = X.shape[0]
        if not(W==None):
            self.dataWeights=W
        else:
            self.dataWeights=np.ones([self.N])
        self.forestType=forestType
    def adaBoost(self,X,Y):
        for i in range(self.nClassifiers):
            print "training classifier %i"%(i)
            if self.forestType=="stump":
                newClassifier = DecisionStump(np.random.randint(0,self.dim))
            else:
                newClassifier = DecisionTree(np.random.randint(0,self.dim))
            newClassifier.trainGreedy(X,Y,self.dataWeights)
            werr=newClassifier.weightedError(X,Y,self.dataWeights)
            alpha = np.log((1.-werr)/werr)
            self.classifiers.append(newClassifier)
            self.classifierWeights.append(alpha)
            for i in range(self.N):
                yhat = newClassifier.evalDefault(X[i])
                if not yhat==Y[i]:
                    self.dataWeights[i] *= (1.-werr)/werr
    def evaluate(self,x):
        r=0
        for i in range(self.nClassifiers):
            r += self.classifiers[i].evalDefault(x) * self.classifierWeights[i]
        # return int(np.sign(r))
        return r
    def __str__(self):
        s=""
        for i in range(self.nClassifiers):
            s+="%.3f %s"%(self.classifierWeights[i], str(self.classifiers[i]))

