#!/usr/bin/env python

import numpy as np
import ROOT
from sys import exit,stderr,stdout
from math import isnan

loadClassifier=True
saveClassifier=False

listOfRawVars = ["fjet1QGtagSub1","fjet1QGtagSub2","fjet1QGtag","fjet1PullAngle","fjet1Pull","fjet1MassTrimmed","fjet1MassPruned","fjet1MassSDbm1","fjet1MassSDb2","fjet1MassSDb0","fjet1QJetVol","fjet1C2b2","fjet1C2b1","fjet1C2b0p5","fjet1C2b0p2","fjet1C2b0","fjet1Tau2","fjet1Tau1","2*fjet1QGtagSub2+fjet1QGtagSub1","fjet1Tau2/fjet1Tau1"]   
nRawVars = len(listOfRawVars)
sigFile = ROOT.TFile("../signal_word.root")
bgFile = ROOT.TFile("../background_word.root")
sigTree = sigFile.Get("DMSTree")
bgTree = bgFile.Get("DMSTree")

ROOT.TMVA.Tools.Instance()
fout = ROOT.TFile("vtagBDT.root","RECREATE")
factory = ROOT.TMVA.Factory("TMVAClassificationCategory", fout,
                            ":".join([
                                "!V",
                                "!Silent",
                                "Color",
                                "DrawProgressBar",
                                "Transformations=I;N",
                                "AnalysisType=Classification"]
                                     ))
for var in listOfRawVars:
  factory.AddVariable(var,'F')
factory.AddSignalTree(sigTree)
factory.AddBackgroundTree(bgTree)

sigCut = ROOT.TCut("abs(fjet1PartonId)==24")
bgCut = ROOT.TCut("")
factory.PrepareTrainingAndTestTree(sigCut,bgCut,
                                    ":".join([
                                      "nTrain_Signal=0",
                                      "nTrain_Background=0",
                                      "SplitMode=Alternate",
                                      "NormMode=NumEvents",
                                      "!V"
                                      ]))

bdtOpts="!H:!V:NTrees=400:BoostType=Grad:Shrinkage=0.1:UseBaggedGrad=F:nCuts=2000:NNodesMax=10000:MaxDepth=5:UseYesNoLeaf=F:nEventsMin=200"
factory.BookMethod(ROOT.TMVA.Types.kBDT, "BDT",bdtOpts)
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
fout.Close()
