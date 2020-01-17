from ae_measure2 import read_ae_file2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

class EqualBinSpace:
    def __init__(self,nbins):
        self.nbins = nbins
    def get_bins(self,sig):
        #return getBreakpoints(self.nbins)
        return np.linspace(min(sig),max(sig),self.nbins+1)
    def to_vect(self,heatmap):
        vect = []
        #get the diagonal
        for i in range(self.nbins):
            vect.append(heatmap[i][i])
        #get just above the diagonal
        for i in range(self.nbins-1):
            vect.append(heatmap[i][i+1])
        return vect

class GaussBinSpace:
    def __init__(self,nbins):
        self.nbins = nbins
    def get_bins(self,sig):
        #return getBreakpoints(self.nbins)
        return np.linspace(min(sig),max(sig),self.nbins+1)
    def to_vect(self,heatmap):
        vect = []
        #get the diagonal
        for i in range(self.nbins):
            vect.append(heatmap[i][i])
        #get just above the diagonal
        for i in range(self.nbins-1):
            vect.append(heatmap[i][i+1])
        return vect

class EquiBinSpace:
    def __init__(self,nbins):
        self.nbins = nbins
    def get_bins(self,sig):
        #return getBreakpoints(self.nbins)
        bins = [np.percentile(sig,(100*i)/self.nbins) 
        for i in range(self.nbins+1)]
        return bins
    def to_vect(self,heatmap):
        vect = []
        #get the diagonal
        for i in range(self.nbins):
            vect.append(heatmap[i][i])
        #get just above the diagonal
        for i in range(self.nbins-1):
            vect.append(heatmap[i][i+1])
        return vect

def sax_normalize(x):
    '''Because SAX says to normalize mean and variance'''
    x2 = x - np.average(x)
    x3 = x2/np.std(x2)
    return x3
'''
def getBreakpoints(k): #k is size of alphabet
    alpha = 1/k #probability of each region
    beta = np.zeros(k-1)
    for i in range(k-1):
        beta[i] = st.norm.ppf(alpha*(i+1))
    return beta
'''
def getBreakpoints(k): #k is size of alphabet
    alpha = 1/k #probability of each region
    beta = np.zeros(k-1)
    for i in range(k-1):
        beta[i] = st.norm.ppf(alpha*(i+1))
    beta2 = np.zeros(k+1)
    beta2[1:-1] = beta
    beta2[0] = -1*np.inf
    beta2[-1] = np.inf
    return beta2

def to_word_bins(sig,space):
    bins = space.get_bins(sig)
    word = np.ones(len(sig))*-1
    for i in range(0,len(bins)-1):
        is_bin = (sig >= bins[i])&(sig <= bins[i+1])
        word[np.where(is_bin)] = i
    return word

def word_to_subword_space(word,space):
    heatmap = np.zeros((space.nbins,space.nbins))
    for i in range(len(word)-1):
        row = int(word[i])
        col = int(word[i+1])
        heatmap[row][col] +=1
    return heatmap/(np.sum(heatmap))

def isnormaldist(x):
    k2, p = stats.normaltest(sax_normalize(v1[0]))
    alpha = 1e-3
    return alpha < p


'''
v1, v2 = read_ae_file2("/home/nolan/Desktop/gg.txt")
NBINS = 5
space = EqualBinSpace(NBINS)
avg1 = np.zeros((NBINS,NBINS))
#get average fibercrack matrix from 
for i in range(0,25):
    sig1 = v1[i]
    sig2 = v2[i]
    sig = sig2
    if max(np.abs(sig1)) > max(np.abs(sig2)):
        sig = sig1
    word = to_word_bins(sig,space)
    heatmap = word_to_subword_space(word,space)
    avg1+= heatmap

avg1 = avg1/25

avg2 = np.zeros((NBINS,NBINS))
for i in range(-2,-22,-1):
    sig1 = v1[i]
    sig2 = v2[i]
    sig = sig2
    if max(np.abs(sig1)) > max(np.abs(sig2)):
        sig = sig1
    word = to_word_bins(sig,space)
    heatmap = word_to_subword_space(word,space)
    avg2+= heatmap
 
avg2 = avg2/20

#avg1[4][4] = 0
#avg2[4][4] = 0

#test data
isf = 0
ism = 0
for i in range(25,43):
    sig1 = v1[i]
    sig2 = v2[i]
    sig = sig2
    if max(np.abs(sig1)) > max(np.abs(sig2)):
        sig = sig1
    word = to_word_bins(sig,space)
    heatmap = word_to_subword_space(word,space)

    matrix_crack = np.sum((avg1 - heatmap)**2)
    fiber_break = np.sum((avg2 - heatmap)**2)
    if matrix_crack > fiber_break:
        ism+=1
    else:
        isf +=1

print('matrix prediction sucess',ism/(ism+isf))

for i in range(-22,-40,-1):
    sig1 = v1[i]
    sig2 = v2[i]
    sig = sig2
    if max(np.abs(sig1)) > max(np.abs(sig2)):
        sig = sig1
    word = to_word_bins(sig,space)
    heatmap = word_to_subword_space(word,space)

    matrix_crack = np.sum((avg1 - heatmap)**2)
    fiber_break = np.sum((avg2 - heatmap)**2)
    if matrix_crack > fiber_break:
        ism+=1
    else:
        isf +=1

print('matrix prediction sucess',isf/(ism+isf))


'''

