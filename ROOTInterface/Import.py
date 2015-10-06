import numpy as np
from ROOT import TFile, TTree, TH1F, gPad
from sys import exit
import multiprocessing as mp
from time import sleep

class TreeImporter(object):
  """kinda like TChain, but for numpy arrays.
  		also does simple computations on the fly
  		(e.g. tau3/tau2, ln(chi), max(sjBtag) etc.)"""
  def __init__(self, tfile,treeName):
    self.treeName = treeName
    self.varList = []
    self.computedVars = []
    self.dependencies = []
    self.counter = 0
    self.goodEntries = None
    self.useGoodEntries = False
    if type(tfile)==type(''):
        self.fIn = TFile(tfile)
    else:
        self.fIn = tfile
    self.tree = self.fIn.FindObjectAny(treeName)
  def draw(self,varName,range=(0,250),cutString=""):
    self.tree.Draw("%s>>htmp(100,%i,%i)"%(varName,range[0],range[1]),cutString)
    htmp = gPad.GetPrimitive("htmp")
    return htmp
  def clone(self,f,t):
    newImporter = TreeImporter(f,t)
    newImporter.resetCounter(self.counter)
    newImporter.goodEntries = self.goodEntries
    for v in self.varList:
      newImporter.addVar(v)
    for v in self.computedVars:
      newImporter.addComputedVar(v)
    return newImporter
  def resetVars(self):
    self.varList = []
    self.computedVars = []
    self.dependencies = []
  def setGoodEntries(self,g):
    self.goodEntries = g
    self.useGoodEntries = True
    self.counter=0
  def resetCounter(self,c=0):
    self.counter = c
    self.goodEntries = None
    self.useGoodEntries = False
  def addVar(self,var):
    self.varList.append(var)
    self.dependencies.append(var)
  def addComputedVar(self,p):
    # p[1] should be a list of variable names in the tree
    # and p[0](p[1]) should be the desired computed variable
    self.computedVars.append((p[0],p[1]))
    self.dependencies += p[1]
  def loadTreeMultithreaded(self,truthValue,nEvents=-1,nProc=4):
    leaves = self.tree.GetListOfLeaves()
    branchDict = {}
    for i in xrange(leaves.GetEntries()):
      leaf = leaves.At(i)
      if leaf.GetName() in self.dependencies:
        branchDict[leaf.GetName()] = leaf
    # figure out which events to load
    if nEvents<0:
      nEvents = np.inf
    nEvents = min(nEvents,self.tree.GetEntries()-self.counter)
    nEventsPerJob = float(nEvents)/nProc
    if not (nEventsPerJob==int(nEventsPerJob)):
      nEventsPerJob = int(nEventsPerJob)+1
    else:
      nEventsPerJob = int(nEventsPerJob)
    # setup multiprocessing stuff
    jobs = []
    manager = mp.Manager()
    q = manager.dict()
    for iJ in xrange(nProc):
      jobs.append(mp.Process(target=self.__coreLoadTree, args=(truthValue,nEventsPerJob,nEventsPerJob*iJ+self.counter,branchDict,q)))
    for j in jobs:
      j.start()
      sleep(10)
    first = True
    for j in jobs:
      if first:
        j.join()
        first=False
      else:
        j.join()
    # combine result of output
    xVals = []
    yVals = []
    for iJ in xrange(nProc):
      vals = q[nEventsPerJob*iJ+self.counter]
      # print 'hello'
      # print vals
      # vals = q.get()
      # if not vals:
      #   print "job did not finish, increase wait time!"
      #   exit(-1)
      xVals.append(vals[0])
      yVals.append(vals[1])
    return np.vstack(xVals),np.hstack(yVals)
  def loadTree(self,truthValue,nEvents=-1):
    leaves = self.tree.GetListOfLeaves()
    branchDict = {}
    for i in xrange(leaves.GetEntries()):
      leaf = leaves.At(i)
      if leaf.GetName() in self.dependencies:
        branchDict[leaf.GetName()] = leaf
    # figure out which events to load
    if nEvents<0:
      nEvents = np.inf
    nEvents = min(nEvents,self.tree.GetEntries()-self.counter)
    return self.__coreLoadTree(truthValue,nEvents,self.counter,branchDict)
  def __coreLoadTree(self,truthValue,nEvents,counter,branchDict,queue=None):
    nEvents = min(nEvents,self.tree.GetEntries()-counter)
    # allocate space
    dataX = np.empty([nEvents,len(self.varList)+len(self.computedVars)])
    dataY = np.ones(nEvents) if truthValue==1 else np.zeros(nEvents) # faster than multiplying if only {0,1}
    # get iterator
    entryIter = xrange(0,nEvents)
    for iE in entryIter:
      # print counter,iE+counter
      self.tree.GetEntry(iE+counter)
      m = 0
      isGood = True
      for var in self.varList:
        dataX[iE,m] = branchDict[var].GetValue()
        if np.isnan(dataX[iE,m]) or np.isinf(dataX[iE,m]):
          dataX[iE,m] = -1 # none of the raw variables have -1 as a real value - but what about storing lnchi now? or maxSubjetBtag?
        m+=1
      for cVar in self.computedVars:
        func = cVar[0]
        xs = cVar[1]
        vals = []
        for name in xs:
          vals.append(branchDict[name].GetValue())
        dataX[iE,m] = func(vals)
        if np.isnan(dataX[iE,m]) or np.isinf(dataX[iE,m]):
          dataX[iE,m] = -99 # sufficiently different from ln chi
        m+=1
    if not(queue==None):
      queue[counter] = (dataX,dataY)
      # queue.put((dataX,dataY))
    else:
    	return dataX,dataY

