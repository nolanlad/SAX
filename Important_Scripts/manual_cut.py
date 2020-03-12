'''
Author: Caelin Muir
Contact: muir@ucsb.edu
Version: 200310

*Cries in manual*
'''

#Imports
from SAX import *
from sklearn.cluster import KMeans
import glob
import sklearn
import numpy as np
import pylab as pl
import matplotlib.ticker as ticker
import os
from ae_measure2 import *
import pandas as pd
from AIC_picker import *

# initialize number of bins (i.e alphabet cardinality), binning scheme, and others
NBINS = 5
space = PercentileBinSpace(NBINS) # have considered equiprobable, it doesn't work



'''
Read-in and Setup
'''
filterb1025 = glob.glob("./b1025_mask_v3.csv")[0]
rawb1025 = glob.glob("./Raw_Data/2_sensor/HNSB2-1_1025.txt")[0]



csv_b1025 = pandas.read_csv(filterb1025)
time = np.array(csv_b1025.Time)
start = np.array(csv_b1025.Start)
end = np.array(csv_b1025.End)

b_v1, b_v2, b_ev = filter_ae(rawb1025, filterb1025)
'''
Snipping Routine
'''
bsnip = []
new_time = []
new_ev = []
for i in range(len(b_ev)):
    if start[i]!=-1:
        bsnipholder = b_v1[i]
        bsnipholder = bsnipholder[int(start[i]):int(end[i])]
        bsnip.append(bsnipholder)
        new_time.append(time[i])
        new_ev.append(b_ev[i])
time = np.array(new_time)
b_ev = np.array(new_ev)



sig1 = b_v1[136][start[136]:end[136]]/np.max(b_v1[136][start[136]:end[136]])
sig2 = b_v1[-1][start[-1]+9:end[-1]]/np.max(b_v1[-1][start[-1]+9:end[-1]])
sig3 = b_v1[0][start[0]:end[0]]/np.max(b_v1[0][start[0]:end[0]])
sig3 = b_v1[59][start[59]:end[59]]/np.max(b_v1[59][start[59]:end[59]])

fingerprint = get_fingerprint(sig3, space)
print(fingerprint)
pl.imshow(fingerprint, cmap = 'Blues')
pl.show()




#pl.plot(sig1)
#pl.plot(sig2)
#pl.show()

#endfingerprint = get_fingerprint()

'''
for i in range(len(b_ev)):
    pl.title('Event ' + str(i))
    pl.plot(b_v1[i])
    pl.show()
'''
