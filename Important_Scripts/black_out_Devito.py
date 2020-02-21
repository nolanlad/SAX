'''
Author: Caelin Muir
Contact: muir@ucsb.edu
Version: 200124

This script generates a Devito analysis where the highest frequency transition
(usually c-c, somtimes b-b or d-d) is set to 0. The idea is similar to diffraction
experiments in which the incident beam is blocked, so the diffraction spots can
be viewed easier. In this case the transition that occurs with the highest transition
is assumed to be a result of Guassian noise pre and post wave.

Plots are saved to same path as experiments.

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
from ae_measure2 import *
from scipy.cluster.vq import whiten
import os



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
space = FullBinSpace(NBINS) # have considered equiprobable, it doesn't work
min_cluster = 2
max_cluster = 12

# Read in file set
batch1_fns = glob.glob("./Raw_Data/VTE_2/*.txt")
print(batch1_fns)


for f in batch1_fns:

    v1, v2, ev= read_ae_file2(f)
    X = get_vect(v1,v2,space)

    silh = np.array([]) # holder arrays
    db_score = np.array([])



    '''
    Blackout
    '''
    for i in range(len(X)):
        X[i][np.argmax(X[i])]=0




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





# Read in single file
batch1_fns = glob.glob("./Raw_Data/VTE_2/HNS2_0826.txt")
v1,v2, ev = read_ae_file2(batch1_fns[0])
X = get_vect(v1,v2,space)




'''
Blackout
'''
for i in range(len(X)):
    X[i][np.argmax(X[i])]=0


# Cluster waveform
kmeans = KMeans(n_clusters=4, n_init=100, tol=1e-6).fit(X)
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
pl.title('Blackout Devito', fontsize=BIGGER_SIZE)
fig.tight_layout()  # otherwise the right y-label is slightly clipped

pl.savefig('Blackout_Devito.png')








for i in range(len(v1)):
    # navigate to appropriate directory
    if lads[i] == 0:
        os.chdir(home+path1)
    elif lads[i] == 1:
        os.chdir(home+path2)
    elif lads[i] == 2:
        os.chdir(home+path3)
    elif lads[i] == 3:
        os.chdir(home+path4)
    else:
        raise ValueError('A very bad thing happened here.')

    # save fingerprint in directory
    fingerprint = get_fingerprint(max_sig(v1[i], v2[i]), space)

    # set max to 0. Needs to be done seperately from vector to fingerprint
    max = argmax2d(fingerprint)
    fingerprint[max[0]][max[1]]=0

    name = 'Event_'+str(i)+'.png'

    fingerprint = upscale(fingerprint) #upscales image
    pl.imsave(name, fingerprint, cmap='Blues')
