#!/usr/bin/env python

import ROOT
import numpy as np
from sys import argv

fin = ROOT.TFile(argv[1])
sh=fin.Get("sh")
bh=fin.Get("bh")
ROOT.gStyle.SetOptStat(0)

c1 = ROOT.TCanvas()
t = ROOT.TLegend(.7,.7,.9,.9)
sh.SetNormFactor()
bh.SetNormFactor()
bh.SetLineColor(2)
bh.GetXaxis().SetTitle("Score")
# bh.GetYaxis().SetTitle("Fraction of events")
bh.SetTitle('')
bh.Draw()
sh.Draw("same")
t.AddEntry(sh,'Signal')
t.AddEntry(bh,'Background')
t.SetFillColor(0)
t.Draw('same')

c1.SaveAs(argv[2])