'''
Author: Caelin Muir
Contact: muir@ucsb.edu
Version: 200127

This script takes in a file path with one experiment as text file
It clusters with k=3 and prints each fingerprint to a folder for its respective label.
Labels of Fingerprints correspond to Danny Don't-Vito.png Note the
initialization scheme defaults to k-means++. Random state is not called.

Fingerprints saved to folders in your home directory named */cluster_i_prints

You have to manually delete the contents of the cluster_i_prints folder prior
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


# set paths for different cluster fingerprints
path1 = '/cluster_1_prints'
path2 = '/cluster_2_prints'
path3 = '/cluster_3_prints'
home = os.getcwd()


# initialize number of bins (i.e alphabet cardinality), binning scheme, and num_clusters (k)
NBINS = 5
space = EqualBinSpace(NBINS)
k = 3

# Read in single file
batch1_fns = glob.glob("./Raw_Data/VTE_2/HNS2_0826.txt")
v1,v2, ev = read_ae_file2(batch1_fns[0])
X = get_heatmaps(v1,v2,space)

# Cluster waveform
kmeans = KMeans(n_clusters=k, n_init=100, tol=1e-6).fit(X)
lads = kmeans.labels_


# Plotting routine for Danny_dont_vito
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

pl.savefig('test.png')







signal = []
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
        raise ValueError('A very bad thing happened here.')

    # save fingerprint in directory
    fingerprint = get_fingerprint(signal[i], space)
    name = 'Event_'+str(i)+'.png'

    fingerprint = upscale(fingerprint) #upscales image
    pl.imsave(name, fingerprint, cmap='Blues')
