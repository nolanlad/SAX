#Imports
from SAX import *
from sklearn.cluster import KMeans
import glob
import sklearn
import numpy as np
import pylab as pl

# initialize number of bins (i.e alphabet cardinality), binning scheme, and others
NBINS = 5
space = EqualBinSpace(NBINS) #Should consider equiprobable bins
max_cluster = 11

# Read in file set, to be used for later
batch1_fns = glob.glob("./Data_from_Amjad/Batch_1/*.txt")


# Read in single file
v1,v2 = read_ae_file2(batch1_fns[0])
X = get_heatmaps(v1,v2,space)


# make heat map of single event
# why did he choose v1, is this automatically the highest amplitude channel, or was this arbitrary?
sig = v1[0]
H = get_fingerprint(sig,space)
pl.imshow(H)
pl.show()
