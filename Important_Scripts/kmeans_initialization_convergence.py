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

# initialize number of bins (i.e alphabet cardinality), binning scheme, and others
NBINS = 5
space = EqualBinSpace(NBINS) # have considered equiprobable, it doesn't work

# Read in file set, to be used for later
batch1_fns = glob.glob("./raw_waveform_txt/VTE_2/*.txt")

# Read in single file
v1,v2 = read_ae_file2(batch1_fns[0])
'''
EBC_0903 from VTE2 was chosen since it had the lowest silhouette value (approx .6),
thus it is the most susceptible to the initialization.
'''
X = get_heatmaps(v1,v2,space)


# Cluster and get stats for 1, 10, 100, and 1000 initializations
nInit = 2 # max number of initializations 10^nInit, 2 => 1,10,100 initializations
n_runs = 100 # number of runs over which the variance is gathered for silh values

inertia_variance = np.array([])
for i in range(nInit+1):
    holder = np.array([])
    for j in range(n_runs):
        kmeans = KMeans(n_clusters=3, n_init=10**(i)).fit(X)
        holder = np.append(holder, kmeans.inertia_)
        #print((i+1)*(j+1))
    inertia_variance = np.append(inertia_variance, np.var(holder))

print(inertia_variance)
pl.title('Convergence of initializations')
pl.xlabel('Number of Initializations')
pl.ylabel('Inertia variance')
pl.semilogx([1,10,100], silh_variance)
pl.show()






'''
pl.title('Convergence of iterations on a cluster initialization')
pl.xlabel('Iteration number')
pl.ylabel('Change in inertia')
pl.legend(["k=3","k=4",'k=5','k=6','k=7','k=8'])
pl.show()
'''
