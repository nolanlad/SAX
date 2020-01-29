#Imports
from SAX import *
from sklearn.cluster import KMeans
import glob
import sklearn
import numpy as np
import pylab as pl
import os

'''
This is Caelin's spaghetti factory
'''

home = os.getcwd()



# initialize number of bins (i.e alphabet cardinality), binning scheme, and others
NBINS = 5
space = EqualBinSpace(NBINS) #Should consider equiprobable bins
max_cluster = 11

# Read in file set, to be used for later
batch1_fns = glob.glob("./VTE_1/HNS2_100418_2-2.txt")


# Read in single file
v1,v2 = read_ae_file2(batch1_fns[0])
X = get_heatmaps(v1,v2,space)


# make heat map of single event
# why did he choose v1, is this automatically the highest amplitude channel, or was this arbitrary?

os.chdir(home)
sig = v1[0]
H = get_fingerprint(sig,space)
#pl.imshow(H, cmap='Blues')



scale = 80
new_data = np.zeros(np.array(H.shape) * scale)
for j in range(H.shape[0]):
    for k in range(H.shape[1]):
        new_data[j * scale: (j+1) * scale, k * scale: (k+1) * scale] = H[j, k]

pl.imsave('test.png', new_data, cmap='Blues')
