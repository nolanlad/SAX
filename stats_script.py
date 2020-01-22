'''
Author: Caelin Muir
Contact: muir@ucsb.edu
Version: 200121

This script takes in a file path with multiple experiments as text files
(ask Bhav how to retrieve this from raw output file), gets the silhouette score
and davies bouldin score for 2 to n cluster. Each cluster is initialized 100
times do avoid falling into local minima
'''

'''
Note the initialization scheme defaults to k-means++. We need to know how this works.
The random state is initialized to be deterministic. WE DO NOT WANT THIS as it means
we are highly likely to fall into a local minima. For this reason we remove the deterministic
seed
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
space = EquiBinSpace(NBINS) #Should consider equiprobable bins
max_cluster = 11

# Read in file set, to be used for later
batch1_fns = glob.glob("./raw_waveform_txt/VTE_2/*.txt")


# Read in single file
v1,v2 = read_ae_file2(batch1_fns[0])
X = get_heatmaps(v1,v2,space)

for f in batch1_fns:

    v1, v2 = read_ae_file2(f)
    X = get_heatmaps(v1,v2,space)

    silh = np.array([])
    db_score = np.array([])

    # Cluster and get stats
    for i in range(2,max_cluster+1):
        kmeans = KMeans(n_clusters=i, n_init=100).fit(X)
        silh = np.append(silh, sklearn.metrics.davies_bouldin_score(X,kmeans.labels_))
        db_score = np.append(db_score, sklearn.metrics.silhouette_score(X,kmeans.labels_))

    # Make and save figures
    pl.plot(range(2,max_cluster+1),silh)
    pl.plot(range(2,max_cluster+1),db_score)
    pl.title(f)
    pl.legend(["silh","D-B"])
    pl.savefig(f[:-4]+'_stats.png')
    pl.clf()
