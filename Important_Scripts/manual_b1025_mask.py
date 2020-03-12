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
import pandas as pd
from AIC_picker import *


# initialize number of bins (i.e alphabet cardinality), binning scheme, and others
NBINS = 5
space = PercentileBinSpace(NBINS) # have considered equiprobable, it doesn't work
k=2
home = os.getcwd()

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
new_ev = []
for i in range(len(b_ev)):
    if start[i]!=-1:
        bsnipholder = b_v1[i]
        bsnipholder = bsnipholder[int(start[i]):int(end[i])]
        bsnip.append(bsnipholder)
        new_time.append(time[i])
        new_ev.append(b_ev[i])
time = np.array(new_time)
b_ev = np.array(new_ev)





'''
b1025
'''
bX = get_vect(bsnip,bsnip,space)



# Cluster waveform
kmeans = KMeans(n_clusters=k, n_init=100, tol=1e-6).fit(bX)
blads = kmeans.labels_


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

plot1 = ax1.scatter(time, blads+1 , color=color1, linewidth=width) #plot silh
pl.title('Danny Don\'t-vito', fontsize=BIGGER_SIZE)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
pl.show()

pl.savefig('mask_b1025.png')

print(len(time))
print(len(time))
df = pd.DataFrame({'Event': b_ev,'Cluster': blads, 'Time': time })
df.to_csv(r'b1025_manual_mask.csv')

pl.clf()
