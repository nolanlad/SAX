'''
Author: Caelin Muir
Contact: muir@ucsb.edu
Version: 200206

This is a working script. See lab book 1, pg.3, Muir
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

# set paths for different clusters
path1 = '/cluster_1_prints'
path2 = '/cluster_2_prints'
path3 = '/cluster_3_prints'
home = os.getcwd()


# initialize number of bins (i.e alphabet cardinality), binning scheme, and others
NBINS = 5
space = EqualBinSpace(NBINS)


# Read in single file, to be used for later
batch1_fns = glob.glob("./Raw_Data/Data_from_Amjad/Fiber_Tow/*.txt")



dist = np.array([])
for f in batch1_fns:
    # Read in single file
    v1,v2,ev = read_ae_file2(f)
    X = get_heatmaps(v1,v2,space)
    fcentroid = np.mean(X, axis=0)

    for i in range (len(X)):
        X[i] = X[i]-fcentroid
        dist = np.append(dist, np.linalg.norm(X[i]))

print(np.var(dist))
print(np.mean(dist))
print(fcentroid)
