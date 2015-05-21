#!/usr/bin/env python

import numpy as np
from array import array
import ROOT
from sys import exit,stderr,stdout
from math import isnan

reader = ROOT.TMVA.Reader()
listOfRawVars = ["DER_mass_MMC","DER_mass_transverse_met_lep","DER_mass_vis","DER_pt_h","DER_deltaeta_jet_jet","DER_mass_jet_jet","DER_prodeta_jet_jet","DER_deltar_tau_lep","DER_pt_tot","DER_sum_pt","DER_pt_ratio_lep_tau","DER_met_phi_centrality","DER_lep_eta_centrality","PRI_tau_pt","PRI_tau_eta","PRI_tau_phi","PRI_lep_pt","PRI_lep_eta","PRI_lep_phi","PRI_met","PRI_met_phi","PRI_met_sumet","PRI_jet_num","PRI_jet_leading_pt","PRI_jet_leading_eta","PRI_jet_leading_phi","PRI_jet_subleading_pt","PRI_jet_subleading_eta","PRI_jet_subleading_phi","PRI_jet_all_pt"]   
nRawVars = len(listOfRawVars)
varDict={}
for var in listOfRawVars:
    varDict[var] = array('f',[0])
    reader.AddVariable(var,varDict[var])
reader.BookMVA("MLP","weights/TMVAClassificationCategory_MLPtanh_Nhidden.weights.xml") # this is higgs with tanh activation
negVal = 0
nSkip = 200000 # > 1 just for development use
print "reading validation data..."
trainYString = np.loadtxt("../training.csv",usecols=[32],skiprows=nSkip,dtype={'names': ('label',),'formats': ('S1',)},delimiter=',')
trainX = np.loadtxt("../training.csv",usecols=range(1,31),delimiter=',',skiprows=nSkip)
trainY = np.array(map( lambda y : 1 if y[0]=='s' else negVal , trainYString))
nerr={}
ntot= {negVal:0,1:0}
testCuts = np. linspace(-1.1,1.1,1000)
for cut in testCuts:
  nerr[cut] = {negVal:0,1:0}
scores=np.empty([len(trainY)])
j=0
for x,y in zip(list(trainX),trainY):
    for i,var in zip(range(nRawVars),listOfRawVars):
        varDict[var][0]=x[i]
    yhat=reader.EvaluateMVA("MLP")
    scores[j]=yhat
    ntot[y]+=1
    for cut in testCuts:
        classifyAs = 1 if yhat > cut else negVal
        if not classifyAs==y:
            nerr[cut][y]+=1
    j+=1

classifierString = "TMVA_MLPtanh_Nhidden"

with open('logs/'+classifierString+'ROC.log','w') as logFile:
  for cut in testCuts:
    logFile.write("%f %f %f\n"%(cut, 1.-float(nerr[cut][1])/ntot[1], float(nerr[cut][negVal])/ntot[negVal]))

with open('logs/'+classifierString+'Scores.log','w') as scoreFile:
  for y,score in zip(trainY,scores):
    scoreFile.write("%i %f\n"%(y,score))