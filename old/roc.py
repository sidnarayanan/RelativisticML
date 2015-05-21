#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
# import sys

nFiles=3
files=[]
label=[]
# files.append('ann/vlogs/vtagBestLargeData1HiddenROC.log')
# label.append('1 hidden layer NN, 1/Z=0.0163')
# files.append('ann/vlogs/vtagBestLargeData2HiddenROC.log')
# label.append('2 hidden layer NN, 1/Z=0.0176')
files.append('ann/vlogs/vtagLargeData3HiddenROC.log')
label.append('3 hidden layer NN, 1/Z=0.0159')
files.append('bdt/vlogs/TMVA_BDT_PASOfficialROC.log')
label.append('CMS Boosted V tagger, 1/Z=0.0155')

# files.append('ann/vlogs/vtagBestWeightedSmallData1HiddenROC.log')
# label.append('Linear output, 1 HL, 1/Z=0.023')
# files.append('ann/vlogs/vtagSigmoidWeightedSmallData1HiddenROC.log')
# label.append('Logistic output, 1 HL, 1/Z=0.024')
# files.append('ann/vlogs/vtagSigmoidWeightedSmallData3HiddenROC.log')
# label.append('Logistic output, 3HL, 1/Z=0.020')

# files.append('ann/logs/higgs3HiddenROC.log')
# label.append('3HL NN, 1/Z=0.042')

# files.append('svm/hlogs/higgsSVMROC.log')
# label.append('RBF SVM, 1/Z=0.186')

# files.append('bdt/vlogs/vtagBDS40ROC.log')
# label.append('40 stumps')
# files.append('bdt/vlogs/vtagBDS60ROC.log')
# label.append('60 stumps')
# files.append('bdt/vlogs/vtagBDS120ROC.log')
# label.append('120 stumps')

files.append('svm/vlogs/vtagSVMROC.log')
label.append('SVM, 1/Z=0.285')

# fig=plt.figure()
# ax=fig.set_yscale('log')
# fig.set_yscale('log')

for n in range(nFiles):
    eff,fr = np.loadtxt(files[n],usecols=(1,2),unpack=True)
    plt.plot(eff,fr,label=label[n])
plt.gca().legend(loc=4)
plt.yscale('log')
plt.xlabel('Signal acceptance rate')
plt.ylabel('Background rejection rate')
plt.show()
plt.savefig('paper/vtagcomp.pdf')

