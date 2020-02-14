'''
Author: Caelin Muir
Contact: muir@ucsb.edu
Version: 200127

This script plots waveforms from all events. I sorted them by hand QQ.
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

home = os.getcwd()
path = '/HNS2_0826_Waveforms'

# Read in single file, to be used for later
batch1_fns = glob.glob("./VTE_2/HNS2_0826.txt")

# Read in single file
v1,v2 = read_ae_file2(batch1_fns[0])

#navigate to save files
os.chdir(home+path)




# Plot those bad boys out!
SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18
width = 2.0


color1 = 'black'
color2 = 'blue'
color3 = 'red'

for i in range(len(v1)):
    sig = max_sig(v1[i],v2[i])
    name = 'Event_' + str(i+1)

    fig, ax1 = pl.subplots()
    ax1.set_ylabel('Voltage (arb. units)', fontsize=MEDIUM_SIZE)
    pl.setp(ax1.get_yticklabels(), visible=False)
    ax1.set_xlabel('Time (arb. units)', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)
    ax1.grid()
    ax1.plot(sig, color=color1, linewidth=width, label='test') #plot silh
    ax1.set_xlim(0,1024)
    pl.title(name, fontsize = BIGGER_SIZE)
    pl.savefig(name+'.png')
    pl.clf()
