
'''
This is Caelin's spaghetti factory
'''

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
from numpy.linalg import norm



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
clust = lads[1]

# Get centroid of cluster of presumed matrix crack
holder = [np.linalg.norm(X[1] - kmeans.cluster_centers_[0]),
    np.linalg.norm(X[1] - kmeans.cluster_centers_[1]),
    np.linalg.norm(X[1] - kmeans.cluster_centers_[2])]
for i in range(3):
    if np.min(holder) == holder[i]:
        centroid = kmeans.cluster_centers_[i]


# Get centroid of cluster of presumed fiber crack
holder = [np.linalg.norm(X[102] - kmeans.cluster_centers_[0]),
    np.linalg.norm(X[102] - kmeans.cluster_centers_[1]),
    np.linalg.norm(X[102] - kmeans.cluster_centers_[2])]
for i in range(3):
    if np.min(holder) == holder[i]:
        fcentroid = kmeans.cluster_centers_[i]

fcentroid = [0.10970674, 0.15978495, 0.41196481, 0.1813392,  0.10800587, 0.0027175,
 0.00434995, 0.00441838, 0.00270772] # centroid of fiber breaks calculated from fiber_variance.py
fib_break = np.array(X[102]) #cast fiber break
matrix_crack = np.array(X[1])


print(np.linalg.norm(fib_break-centroid))
print(np.linalg.norm(fib_break-fcentroid))


#normalize
centroid = centroid/np.linalg.norm(centroid)
fcentroid = fcentroid/np.linalg.norm(fcentroid)


'''
#print(np.inner(centroid, fcentroid))
print(np.inner(centroid, matrix_crack))
print(np.inner(fcentroid, fib_break))
print(np.inner(fcentroid, matrix_crack))
print(np.inner(centroid, fib_break))
'''







'''
home = os.getcwd()



# initialize number of bins (i.e alphabet cardinality), binning scheme, and others
NBINS = 5
space = EqualBinSpace(NBINS) #Should consider equiprobable bins
max_cluster = 11

# Read in file set, to be used for later
batch1_fns = glob.glob("./VTE_1/HNS2_100418_2-2.txt")


# Read in single file
v1,v2 = read_ae_file2(batch1_fns[0])
X = get_heatmaps(v1,v2,space)


# make heat map of single event
# why did he choose v1, is this automatically the highest amplitude channel, or was this arbitrary?

os.chdir(home)
sig = v1[0]
H = get_fingerprint(sig,space)
#pl.imshow(H, cmap='Blues')
print(type(sig))


scale = 80
new_data = np.zeros(np.array(H.shape) * scale)
for j in range(H.shape[0]):
    for k in range(H.shape[1]):
        new_data[j * scale: (j+1) * scale, k * scale: (k+1) * scale] = H[j, k]

pl.imsave('test.png', H, cmap='Blues')
'''
