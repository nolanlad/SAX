'''
Author: Caelin Muir
Contact: muir@ucsb.edu
Version: 200127

This script takes in a file path with one experiment as text files
(ask Bhav how to retrieve this from raw output file). It clusters into 3 labels,
and prints each fingerprint to a folder for its respective label. Labels of Fingerprints
correspond to Danny Don't-Vito.png

Note the initialization scheme defaults to k-means++. Random state is not called.
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
space = EqualBinSpace(NBINS) # have considered equiprobable, it doesn't work
min_cluster = 2
max_cluster = 12

# Read in single file, to be used for later
batch1_fns = glob.glob("./VTE_1/HNS2_100418_2-2.txt")

# Read in single file
v1,v2 = read_ae_file2(batch1_fns[0])
X = get_heatmaps(v1,v2,space)

# Set empty stat array
silh = np.array([])
db_score = np.array([])

# Cluster waveform and get stats
kmeans = KMeans(n_clusters=3, n_init=100, tol=1e-6).fit(X)
db_score = np.append(db_score, sklearn.metrics.davies_bouldin_score(X,kmeans.labels_))
silh = np.append(silh, sklearn.metrics.silhouette_score(X,kmeans.labels_))

lads = kmeans.labels_


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



# Fingerprints of Danny Don't Vito clusters

# generate empty signal holder
signal = []

# Generate set of signals with highest intensity
for i in range(len(v1)):
    if max(abs(v1[i])) > max(abs(v2[i])):
        signal.append(v1[i])
    else:
        signal.append(v2[i])
signal = np.asarray(signal)


for i in range(len(signal)):
    # navigate to appropriate directory
    if lads[i] == 0:
        os.chdir(home+path1)
    elif lads[i] == 1:
        os.chdir(home+path2)
    elif lads[i] == 2:
        os.chdir(home+path3)
    else:
        raise ValueError('A very bad thing happened here. A signal clustered outside standard 3')

    # save fingerprint in directory
    fingerprint = get_fingerprint(signal[i], space)
    name = 'Event_'+str(i)+'.png'

    fingerprint = upscale(fingerprint) #upscales image
    pl.imsave(name, fingerprint, cmap='Blues')
