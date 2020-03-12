'''
Author: Caelin Muir
Contact: muir@ucsb.edu
Version: 200312
This script takes in a file path with multiple experiments as text files
gets the silhouette score and davies bouldin score for min_cluster to
max_cluster cluster. Each cluster is initialized 100 times do avoid falling
into local minima.
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
from AIC_picker import *

# initialize number of bins (i.e alphabet cardinality), binning scheme, and others
NBINS = 3
space = PercentileBinSpace(NBINS) # have considered equiprobable, it doesn't work
min_cluster = 2
max_cluster = 12

# Read in file set
'''
Read-in and Setup
'''
filterb1025 = glob.glob("./b1025_mask_v3.csv")[0]
rawb1025 = glob.glob("./Raw_Data/2_sensor/HNSB2-1_1025.txt")[0]



csv_b1025 = pandas.read_csv(filterb1025)
time = np.array(csv_b1025.Time)
start = np.array(csv_b1025.Start)
end = np.array(csv_b1025.End)

b_v1, b_v2, b_ev = filter_ae(rawb1025, filterb1025)

'''
Snipping Routine
'''
bsnip = []
new_time = []
for i in range(len(b_ev)):
    if start[i]!=-1:
        bsnipholder = b_v1[i]
        bsnipholder = bsnipholder[int(start[i]):int(end[i])]
        bsnip.append(bsnipholder)
        new_time.append(time[i])
time = np.array(new_time)




X = get_vect(bsnip, bsnip, space)


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

pl.show()
