'''
Author: Caelin Muir
Contact: muir@ucsb.edu
Version: 200122

This script is a sanity check to determine the appropriate number of iterations
needed for a given kMeans initialization to converge for the k-means++ initialization
scheme. Others should be tested
'''


#Imports
from SAX import *
from sklearn.cluster import KMeans
import glob
import sklearn
import numpy as np
import pylab as pl

# list of all inertia values as input, outputs change in inertia from i to i+1
def get_tolerance(inertia):
    tolerance = np.array([])
    for i in range(len(inertia)-1):
        tolerance = np.append(tolerance, inertia[i]-inertia[i+1])
    return tolerance

# initialize number of bins (i.e alphabet cardinality), binning scheme, and others
NBINS = 5
space = EqualBinSpace(NBINS) # have considered equiprobable, it doesn't work

# Read in file set, to be used for later
batch1_fns = glob.glob("./raw_waveform_txt/VTE_2/*.txt")

# Read in single file
print(batch1_fns[-2])
v1,v2 = read_ae_file2(batch1_fns[0])
X = get_heatmaps(v1,v2,space)

# ayyy
lmao = 42069

# Cluster and get stats, non-random seed is ok for this purpose
for j in range(6):
    inertia = np.array([])
    for i in range(15): #note the +3 comes from starting at 3
        kmeans = KMeans(n_clusters=j+3, random_state=lmao, n_init=1,  max_iter=i+1, tol=1e-60).fit(X)
        inertia = np.append(inertia, kmeans.inertia_)
    pl.plot(get_tolerance(inertia))

pl.title('Convergence of iterations on a cluster initialization')
pl.xlabel('Iteration number')
pl.ylabel('Change in inertia')
pl.legend(["k=3","k=4",'k=5','k=6','k=7','k=8'])
pl.show()
