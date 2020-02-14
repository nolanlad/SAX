'''
Author: Caelin Muir
Contact: muir@ucsb.edu
Version: 200130

Library for my wavelet decomposition scheme following Emmanuels paper

Note the initialization scheme defaults to k-means++. Random state is not called.

'''

import pywt as wave
from sklearn.cluster import KMeans
import glob
import sklearn
import numpy as np
import pylab as pl
import matplotlib.ticker as ticker
from ae_measure2 import *
from SAX import *

# import signal
batch1_fns = glob.glob("./VTE_2/HNS2_0826.txt")
v1,v2 = read_ae_file2(batch1_fns[0])
signal = max_sig(v1[0], v2[0])


w = wave.Wavelet('db1')

tree = wave.wavedec(signal, w, level=1)

print(tree[0])
pl.plot(tree[0])
pl.show()
