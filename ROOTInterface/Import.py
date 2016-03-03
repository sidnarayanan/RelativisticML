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
    self.cuts = []
    self.dependencies = []
    self.counter = 0
    self.goodEntries = []
    self.useGoodEntries = False
    self.treeName = treeName
    if type(tfile)==type(''):
        self.fIn = TFile(tfile)
    else:
        self.fIn = tfile
    self.tree = self.fIn.FindObjectAny(treeName)
  def addFriend(self,f):
    friend = self.fIn.FindObjectAny(f)
    self.tree.AddFriend(friend)
  # def draw(self,varName,range=(0,250),cutString=""):
  #   self.tree.Draw("%s>>htmp(100,%i,%i)"%(varName,range[0],range[1]),cutString)
  #   htmp = gPad.GetPrimitive("htmp")
  #   return htmp
  def clone(self,f,t):
    newImporter = TreeImporter(f,t)
    newImporter.resetCounter(self.counter)
    for v in self.varList:
      newImporter.addVar(v)
    for v in self.computedVars:
      newImporter.addComputedVar(v)
    for c in self.cuts:
      newImporter.addCut(c)
    return newImporter
  def addCut(self,c):
    # c[1] should be a list of variables name in the tree
    # and c[0](c[1]) should return true for passing events
    self.cuts.append(c)
    for v in c[1]:
      self.dependencies.append(v)
  def resetVars(self):
    self.varList = []
    self.computedVars = []
    self.cuts = []
    self.dependencies = []
  def resetCounter(self,c=0):
    self.counter = c
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
    q = manager.dict() # this is not really a dict and I am offline so cannot look up documentation for DictProxy
    for iJ in xrange(nProc):
      offset = nEventsPerJob*iJ+self.counter
      jobs.append(mp.Process(target=self.__coreLoadTree, args=(truthValue,nEventsPerJob,offset,branchDict,q)))
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
      xVals.append(vals[0])
      yVals.append(vals[1])
    return np.vstack(xVals),np.hstack(yVals)
  def loadTree(self,truthValue,nEvents=-1):
    leaves = self.tree.GetListOfLeaves()
    branchDict = {}
    print self.dependencies
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
    # if queue:
    #   # this is a multithreaded go
    #   fIn = TFile(self.fileName)
    #   tree = fIn.GetObjectAny(self.treeName)
    #   leaves = self.tree.GetListOfLeaves()
    #   branchDict = {}
    #   for i in xrange(leaves.GetEntries()):
    #     leaf = leaves.At(i)
    #     if leaf.GetName() in self.dependencies:
    #       branchDict[leaf.GetName()] = leaf
    nEvents = min(nEvents,self.tree.GetEntries()-counter)
    # allocate space
    dataX = np.empty([nEvents,len(self.varList)+len(self.computedVars)])
    dataY = np.ones(nEvents) if truthValue==1 else np.zeros(nEvents) # faster than multiplying if only {0,1}
    # get iterator
    if self.useGoodEntries:
      entryIter = self.goodEntries
    else:
      entryIter = xrange(0,nEvents)
    nEntries=0
    for iE in entryIter:
      # print counter,iE+counter
      self.tree.GetEntry(iE+counter)
      m = 0
      isGood = True
      for cut in self.cuts:
        vals = []
        for name in cut[1]:
          vals.append(branchDict[name].GetValue())
        if not(cut[0](vals)):
          isGood = False
          break
      if not isGood:
        continue
      if not self.useGoodEntries:
        self.goodEntries.append(iE+counter)
      for var in self.varList:
        try:
          dataX[nEntries,m] = branchDict[var].GetValue()
        except KeyError:
          dataX[nEntries,m] = getattr(self.tree,var) # I think this is slower
        if np.isnan(dataX[nEntries,m]) or np.isinf(dataX[nEntries,m]):
          dataX[nEntries,m] = -1 # none of the raw variables have -1 as a real value  # but what about storing lnchi now? or maxSubjetBtag?
        m+=1
      for cVar in self.computedVars:
        func = cVar[0]
        xs = cVar[1]
        vals = []
        for name in xs:
          vals.append(branchDict[name].GetValue())
        dataX[nEntries,m] = func(vals)
        if np.isnan(dataX[nEntries,m]) or np.isinf(dataX[nEntries,m]):
          dataX[nEntries,m] = -99 # sufficiently different from ln chi
        m+=1
      nEntries+=1
    dataX = dataX[:nEntries]
    dataY = dataY[:nEntries]
    if not(queue==None):
      queue[counter] = (dataX,dataY)
      # queue.put((dataX,dataY))
    else:
    	return dataX,dataY

