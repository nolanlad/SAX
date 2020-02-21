'''
Author: Caelin Muir
Contact: muir@ucsb.edu
Version: 200127

This script takes in a file path with one experiment as text files
(ask Bhav how to retrieve this from raw output file). It clusters into 3 labels,
and prints each fingerprint to a folder for its respective label. Labels of Fingerprints
correspond to Danny Don't-Vito.png

Note the initialization scheme defaults to k-means++. Random state is not called.


Also, you have to manually delete the contents of the cluster_i_prints folder prior
to running, otherwise you will get prints from the last run mixed in.
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
from scipy.cluster.vq import whiten


# set paths for different clusters
path1 = '/cluster_1_prints'
path2 = '/cluster_2_prints'
path3 = '/cluster_3_prints'
home = os.getcwd()


min_cluster = 2
max_cluster = 12


# initialize number of bins (i.e alphabet cardinality), binning scheme, and others
NBINS = 5
space = FullBinSpace(NBINS) # have considered equiprobable, it doesn't work

# Read in single file, to be used for later
batch1_fns = glob.glob("./Raw_Data/VTE_2/HNS2_0826.txt")

# Read in single file
v1,v2, ev = read_ae_file2(batch1_fns[0])
X = get_vect(v1,v2,space)




'''
FIRST CLUSTERING TASK
'''
# Set empty stat array
silh = np.array([])
db_score = np.array([])

# Cluster waveform and get stats
kmeans = KMeans(n_clusters=3, n_init=100, tol=1e-6).fit(X)
lads = kmeans.labels_



# get densest cluster
counts = np.bincount(lads)
dense_clust = np.argmax(counts)


Y=[]
for i in range(len(lads)):
    if lads[i]==dense_clust:
        Y.append(X[i])






'''
SECOND CLUSTERING TASK (get stats within densest cluster)
'''


silh = np.array([])
db_score = np.array([])


# Cluster and get stat
for i in range(min_cluster, max_cluster+1):
    kmeans = KMeans(n_clusters=i, n_init=100, tol=1e-6).fit(Y)
    db_score = np.append(db_score, sklearn.metrics.davies_bouldin_score(Y,kmeans.labels_))
    silh = np.append(silh, sklearn.metrics.silhouette_score(Y,kmeans.labels_))



'''Plot stats of second clustering task'''
SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18
width = 2.0

fig, ax1 = pl.subplots()
color1 = 'black'
color2 = 'blue'
color3 = 'red'

# ax1.set_xlabel('Number of clusters')
ax1.set_ylabel('Silhouette Score', fontsize=MEDIUM_SIZE)
ax1.tick_params(axis='y', labelcolor=color1, labelsize = MEDIUM_SIZE)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax1.set_xlabel('Number of clusters', fontsize=MEDIUM_SIZE)
ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)
ax1.set_xlim(min_cluster, max_cluster)
ax1.grid()
plot1 = ax1.plot(range(2,max_cluster+1), silh , color=color1, linewidth=width, label='Silhouette Score') #plot silh

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Davies Bouldin Score',fontsize=MEDIUM_SIZE)
ax2.tick_params(axis='y', labelcolor=color1, labelsize=MEDIUM_SIZE)
plot2 = ax2.plot(range(2,max_cluster+1), db_score, color=color2, linewidth=width, label='Davies-Bouldin') #plot db score

pl.title('Cluster Statisics', fontsize=BIGGER_SIZE)

heh = plot1+plot2
labs = [l.get_label() for l in heh]
ax2.legend(heh, labs, loc='upper right')


fig.tight_layout()  # otherwise the right y-label is slightly clipped


os.chdir(home)
pl.savefig('Densestcluster_stats.png')
pl.clf()






















# Plot those bad boys out! i.e. danny don't vito
SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18
width = 2.0

fig, ax1 = pl.subplots()
color1 = 'black'
color2 = 'blue'
color3 = 'red'

ax1.set_ylabel('Cluster number', fontsize=MEDIUM_SIZE)
ax1.tick_params(axis='y', labelcolor=color1, labelsize = MEDIUM_SIZE)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))

ax1.set_xlabel('Event Number', fontsize=MEDIUM_SIZE)
ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)

ax1.grid()
plot1 = ax1.scatter(range(len(lads)), lads+1 , color=color1, linewidth=width) #plot silh

pl.title('Danny Don\'t-vito', fontsize=BIGGER_SIZE)
fig.tight_layout()  # otherwise the right y-label is slightly clipped


pl.savefig('Danny_dont_vito.png')
