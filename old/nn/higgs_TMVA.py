#!/usr/bin/env python

import numpy as np
import ROOT
from sys import exit,stderr,stdout
from math import isnan

loadClassifier=True
saveClassifier=False

listOfRawVars = ["DER_mass_MMC","DER_mass_transverse_met_lep","DER_mass_vis","DER_pt_h","DER_deltaeta_jet_jet","DER_mass_jet_jet","DER_prodeta_jet_jet","DER_deltar_tau_lep","DER_pt_tot","DER_sum_pt","DER_pt_ratio_lep_tau","DER_met_phi_centrality","DER_lep_eta_centrality","PRI_tau_pt","PRI_tau_eta","PRI_tau_phi","PRI_lep_pt","PRI_lep_eta","PRI_lep_phi","PRI_met","PRI_met_phi","PRI_met_sumet","PRI_jet_num","PRI_jet_leading_pt","PRI_jet_leading_eta","PRI_jet_leading_phi","PRI_jet_subleading_pt","PRI_jet_subleading_eta","PRI_jet_subleading_phi","PRI_jet_all_pt"]   
nRawVars = len(listOfRawVars)
trainFile = ROOT.TFile("../training.csv.root")
trainTree = trainFile.Get("htautau")

ROOT.TMVA.Tools.Instance()
fout = ROOT.TFile("higgsANN.root","RECREATE")
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
factory.AddSignalTree(trainTree)
factory.AddBackgroundTree(trainTree)

sigCut = ROOT.TCut("Label==1")
bgCut = ROOT.TCut("Label==0")
factory.PrepareTrainingAndTestTree(sigCut,bgCut,
                                    ":".join([
                                      "nTrain_Signal=0",
                                      "nTrain_Background=0",
                                      "SplitMode=Alternate",
                                      "NormMode=NumEvents",
                                      "!V"
                                      ]))

annOpts="!H:!V:NCycles=200:NeuronType=tanh"
factory.BookMethod(ROOT.TMVA.Types.kMLP, "MLPtanh_Nhidden",annOpts)
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
fout.Close()
