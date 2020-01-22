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

'''
Generates equiprobable bin spacings assuming a Gaussian distribution
'''
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


'''
Generates equiprobable bin spacings assuming no probability distribution
'''
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
takes signal and desired spacing strategy and generates your word. 1D vector of voltage values
'''
def to_word_bins(sig,space):
    bins = space.get_bins(sig)
    word = np.ones(len(sig))*-1
    for i in range(0,len(bins)-1):
        is_bin = (sig >= bins[i])&(sig <= bins[i+1])
        word[np.where(is_bin)] = i
    return word


'''
takes a SAX word and generates a fingerprint sliding window size of 2 is hard coded in this function and all others
'''
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
v1, v2 are the different channels, space is defined breakpoints. Classes of these
can be found above
'''
def get_heatmaps(v1,v2,space):
    X = []
    for i in range(0,len(v1)):
        sig1 = v1[i]
        sig2 = v2[i]
        sig = sig2
        if max(np.abs(sig1)) > max(np.abs(sig2)):
            sig = sig1
        word = to_word_bins(sig,space)
        heatmap = word_to_subword_space(word,space)
        X.append(space.to_vect(heatmap))
    return X
