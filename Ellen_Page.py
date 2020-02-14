'''
Author: Caelin Muir
Contact: muir@ucsb.edu
Version: 200127

This script plots waveforms from the 3 different clusters
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



# Read in single file, to be used for later
batch1_fns = glob.glob("./VTE_2/HNS2_0826.txt")

# Read in single file
v1,v2 = read_ae_file2(batch1_fns[0])



# Plot those bad boys out! i.e. danny don't vito
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 18
width = 2.0

color1 = 'black'
color2 = 'blue'
color3 = 'red'


ellen = pl.figure()
ax1 = pl.subplot(311)
ax2 = pl.subplot(312, sharex = ax1)
ax3 = pl.subplot(313, sharex = ax1)

ax1.plot(v1[51], color=color1)
ax2.plot(v1[12], color=color2)
ax3.plot(v1[67], color=color3)

# format ax1
pl.setp(ax1.get_xticklabels(), visible=False)
pl.setp(ax1.get_yticklabels(), visible=False)
ax1.tick_params(axis='x', which='both', length=0)
ax1.set_ylabel('Cluster 1', fontsize=MEDIUM_SIZE)


# format ax2
ax2.set_ylabel('Voltage (arb. units)', fontsize=MEDIUM_SIZE)
pl.setp(ax2.get_xticklabels(), visible=False)
pl.setp(ax2.get_yticklabels(), visible=False)
ax2.set_ylabel('Cluster 2', fontsize=MEDIUM_SIZE)

#fomat ax3
pl.setp(ax3.get_yticklabels(), visible=False)
ax3.set_ylabel('Cluster 3', fontsize=MEDIUM_SIZE)
ax3.set_xlabel('Time (arb. units)', fontsize=MEDIUM_SIZE)
ax3.set_xlim(0,1024)

ax1.grid()
ax2.grid()
ax3.grid()

pl.show()
