import numpy as np
import ROOT as root
from ROOTInterface.Import import TreeImporter

sigFile = root.TFile('/home/sid/scratch/data/signal_AK8fj.root')
importer = TreeImporter(sigFile,'jets')
importer.addVarList(['massSoftDrop'])
X,Y = importer.loadTree(1,10)
print X,Y