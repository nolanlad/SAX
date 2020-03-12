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




'''
Read-in and Setup
'''
s9225_mask = glob.glob("./s9225_mask.csv")[0]
b1025_mask = glob.glob("./b1025_mask.csv")[0]
offset = 2.739



csv = pandas.read_csv(s9225_mask)
sClust = np.array(csv.Cluster)
sTime  = np.array(csv.Time)

csv = pandas.read_csv(b1025_mask)
bClust = np.array(csv.Cluster)
bTime  = np.array(csv.Time)

# relabel b1025
for i in range(len(bClust)):
    if bClust[i] == 0:
        bClust[i] = 2

    elif bClust[i] == 1:
        bClust[i] = 0

    elif bClust[i] == 2:
        bClust[i] = 1



wrong=0
for i in range(len(sClust)):
    if sClust[i] != bClust[i]:
        wrong +=1
print(wrong/len(sClust))

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

ax1.set_xlabel('Time (s)', fontsize=MEDIUM_SIZE)
ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)
ax1.grid()

plot1 = ax1.scatter(sTime, sClust+1, color=color2, linewidth=width, label='s9225', alpha=.3)
plot2 = ax1.scatter(bTime, bClust+1, color=color3, linewidth=width,  label='b1025', alpha=.3)

pl.title('2-1 Mask', fontsize=BIGGER_SIZE)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
pl.show()
