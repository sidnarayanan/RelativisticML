import numpy as np
from ROOT import TFile, TTree, TH1F, gPad
from sys import exit

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
    # allocate space
    dataX = np.empty([nEvents,len(self.varList)+len(self.computedVars)])
    dataY = np.ones(nEvents) if truthValue==1 else np.zeros(nEvents) # faster than multiplying if only {0,1}
    # get iterator
    entryIter = xrange(self.counter,self.counter+nEvents)
    for iE in entryIter:
      self.tree.GetEntry(iE)
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
    return dataX,dataY

