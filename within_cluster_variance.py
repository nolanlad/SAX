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



# initialize number of bins (i.e alphabet cardinality), binning scheme, and others
NBINS = 5
space = EqualBinSpace(NBINS) # have considered equiprobable, it doesn't work

# Read in single file, to be used for later
batch1_fns = glob.glob("./Raw_Data/VTE_2/HNS2_0826.txt")

# Read in single file
v1,v2, ev = read_ae_file2(batch1_fns[0])
X = get_heatmaps(v1,v2,space)

# Cluster waveforms in experiment
kmeans = KMeans(n_clusters=3, n_init=100, tol=1e-6).fit(X)
lads = kmeans.labels_
# ASSUME CLUSTER 3 is fiber breakpoints, this gets that cluster label
clust = lads[102]

# Get centroid of cluster of presumed fiber breaks
holder = [np.linalg.norm(X[102] - kmeans.cluster_centers_[0]),
    np.linalg.norm(X[102] - kmeans.cluster_centers_[1]),
    np.linalg.norm(X[102] - kmeans.cluster_centers_[2])]
for i in range(3):
    if np.min(holder) == holder[i]:
        centroid = kmeans.cluster_centers_[i]




dist = np.array([])
for i in range(len(lads)):
    if lads[i]==clust:
        X[i] = X[i]-centroid # centers cluster of fiber breaks at 0
        dist = np.append(dist, np.linalg.norm(X[i]))


print(np.var(dist))
print(np.mean(dist))
print(np.linalg.norm(centroid-X[102]))

fcentroid = [0.10970674, 0.15978495, 0.41196481, 0.1813392,  0.10800587, 0.0027175,
 0.00434995, 0.00441838, 0.00270772] # centroid of fiber breaks calculated from fiber_variance.py
print(np.linalg.norm(fcentroid-X[102]))
