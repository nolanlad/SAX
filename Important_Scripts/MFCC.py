'''
Author: Caelin Muir
Contact: muir@ucsb.edu
Version: 200124

Assumes all data is normalized
'''

#Imports
from SAX import *
from sklearn.cluster import KMeans
import glob
import sklearn
import numpy as np
import pylab as pl
import matplotlib.ticker as ticker
from ae_measure2 import *
from scipy.cluster.vq import whiten
import os
import sigproc #from github, need to cite if we end up using this.
import base # see above


# Jank
def argmax2d(X):
    n, m = X.shape
    x_ = np.ravel(X)
    k = np.argmax(x_)
    i, j = k // m, k % m
    return i, j


# set paths for different cluster fingerprints
path1 = '/cluster_1_prints'
path2 = '/cluster_2_prints'
path3 = '/cluster_3_prints'
path4 = '/cluster_4_prints'
home = os.getcwd()



# initialize number of bins (i.e alphabet cardinality), binning scheme, and others
NBINS = 5
space = PercentileBinSpace(NBINS) # have considered equiprobable, it doesn't work
min_cluster = 2
max_cluster = 12
dt = 0.0000001
rate = 1/dt
window = dt*1024 #Hardcoded for 1024 data points
ratio = .01/.025*window

# Read in file set
batch1_fns = glob.glob("./Raw_Data/VTE_2/*.txt")

for f in batch1_fns:
    # Get set of signals from 1 experiment with the highest value per channel
    v1, v2, ev= read_ae_file2(f)
    sig=[]
    for i in range(len(v1)):
        sig.append(max_sig(v1[i], v2[i]))
    sig = np.array(sig)

    # jank code that converts raw signal to vector of mfcc
    holder = []
    for i in range(len(sig)):
        holder.append(base.mfcc(sig[i], samplerate=rate, winlen=window,
            winstep=ratio, lowfreq=300000, highfreq=1800000))

    X = []
    for i in range(len(sig)):
        X.append(holder[i][0])



    '''
    CLUSTER STATISTICS ROUTINE
    '''
    silh = np.array([]) # holder arrays
    db_score = np.array([])

    # Cluster and get stat
    for i in range(min_cluster, max_cluster+1):
        kmeans = KMeans(n_clusters=i, n_init=100, tol=1e-6).fit(X)
        db_score = np.append(db_score, sklearn.metrics.davies_bouldin_score(X,kmeans.labels_))
        silh = np.append(silh, sklearn.metrics.silhouette_score(X,kmeans.labels_))

    # Plotting routine
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

    pl.savefig(f[:-4]+'_stats.png')
    pl.clf()
    print('Plot ' + str(f) + ' saved.')
