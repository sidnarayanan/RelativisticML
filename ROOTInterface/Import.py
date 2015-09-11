import numpy as np
from ROOT import TFile, TTree

class TreeImporter(object):
  """kinda like TChain, but for numpy arrays.
  		also does simple computations on the fly
  		(e.g. tau3/tau2, log(chi), etc.)"""
  def __init__(self, tfile,treeName):
    self.treeName = treeName
    self.varList = []
    self.computedVars = []
    self.dependencies = []
    self.counter = 0
    if type(tfile)==type(''):
        self.fIn = TFile(tfile)
    else:
        self.fIn = tfile
    self.tree = self.fIn.FindObjectAny(treeName)
  def clone(self,f,t):
  	newImporter = TreeImporter(f,t)
  	newImporter.resetCounter(self.counter)
  	newImporter.addVarList(self.varList)
  	for v in self.computedVars:
  		newImporter.addComputedVar(v)
  	return newImporter
  def resetVars(self):
    self.varList = []
    self.computedVars = []
    self.dependencies = []
  def resetCounter(self,c=0):
    self.counter = c
  def addVarList(self,varList):
    self.varList += varList
    self.dependencies += varList
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
    if nEvents<0:
      nEvents = np.inf
    nEvents = min(nEvents,self.tree.GetEntries()-self.counter)
    dataX = np.empty([nEvents,len(self.varList)+len(self.computedVars)])
    dataY = np.ones(nEvents)*truthValue
    for iE in xrange(self.counter,self.counter+nEvents):
      self.tree.GetEntry(iE)
      m = 0
      isGood = True
      for var in self.varList:
        dataX[iE,m] = branchDict[var].GetValue()
        if np.isnan(dataX[iE,m]) or np.isinf(dataX[iE,m]):
          dataX[iE,m] = -999 # make it sufficiently different from values of ln(chi)
        m+=1
      for cVar in self.computedVars:
        func = cVar[0]
        xs = cVar[1]
        vals = []
        for name in xs:
          vals.append(branchDict[name].GetValue())
        dataX[iE,m] = func(vals)
        if np.isnan(dataX[iE,m]) or np.isinf(dataX[iE,m]):
          dataX[iE,m] = -999
        m+=1
    return dataX,dataY

