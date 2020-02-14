'''
Author: Nolan McCarthy
Contact: nolanrmccarthy@gmail.com
Version: 200213

This is a class definition and function definition file for SAX clustering routine
as developed by the Daly Lab
'''




from ae_measure2 import read_ae_file2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

class EqualBinSpace:
    '''
    Creates class for SAX scheme with equally spaced breakpoints
    '''
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


class FullBinSpace:
    '''
    Creates class for SAX scheme with equally spaced breakpoints, returns full
    vector to cluster over
    '''
    def __init__(self,nbins):
        self.nbins = nbins
    def get_bins(self,sig):
        #return getBreakpoints(self.nbins)
        return np.linspace(min(sig),max(sig),self.nbins+1)
    def to_vect(self,heatmap):
        vect = []
        for n in range(self.nbins):
            for m in range(self.nbins):
                vect.append(heatmap[n][m])
        return vect


class GaussBinSpace:
    '''
    Class for SAX scheme wtih equiprobable bin spacings assuming a Gaussian distribution
    Note this functionality does not work
    '''
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



class PercentileBinSpace:
    '''
    Class for SAX scheme with equiprobable bin spacings assuming no probability distribution.
    '''
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


def sax_normalize(signal):
    '''
    Normalizes signal to have mean 0 and unit variance

    Signal: array-like
    '''
    x2 = signal - np.average(signal)
    x3 = x2/np.std(signal)
    return x3


#need to rename
def to_word_bins(sig,space):
    '''
    Generates SAX word

    sig =
    takes signal and desired spacing strategy and generates your word. 1D vector of voltage values
    '''
    bins = space.get_bins(sig)
    word = np.ones(len(sig))*-1
    for i in range(0,len(bins)-1):
        is_bin = (sig >= bins[i])&(sig <= bins[i+1])
        word[np.where(is_bin)] = i
    return word


# rename and document
def word_to_subword_space(word,space):
    '''
    takes a SAX word and generates a fingerprint sliding window size of 2 is
    hard coded in this function and all others
    '''
    heatmap = np.zeros((space.nbins,space.nbins))
    for i in range(len(word)-1):
        row = int(word[i])
        col = int(word[i+1])
        heatmap[row][col] +=1
    return heatmap/(np.sum(heatmap))


#document
def isnormaldist(x):
    k2, p = stats.normaltest(sax_normalize(v1[0]))
    alpha = 1e-3
    return alpha < p



# rename and document
def get_heatmaps(v1,v2,space):
    '''
    v1, v2 are the different channels, space is defined breakpoints. Classes of these
    can be found above.
    '''
    X = []
    for i in range(len(v1)):
        sig = max_sig(v1[i], v2[i]) # get highest signal
        word = to_word_bins(sig,space)
        heatmap = word_to_subword_space(word,space)
        X.append(space.to_vect(heatmap))
    return X


#document
def get_fingerprint(sig,space):
    word = to_word_bins(sig,space)
    heatmap = word_to_subword_space(word,space)
    return heatmap


#document/reformat
def upscale(fingerprint, scale=80):
    '''
    Takes in a fingerprint object and upscales it so it can actually be viewed when
    you save and open it >:(

    Note, this is a really hacky way to do it and under any other circumstances,
    i.e. not a bit map, I don't think this would work
    '''
    new_data = np.zeros(np.array(fingerprint.shape) * scale)
    for j in range(fingerprint.shape[0]):
        for k in range(fingerprint.shape[1]):
            new_data[j * scale: (j+1) * scale, k * scale: (k+1) * scale] = fingerprint[j, k]
    return new_data



def max_sig(signal1, signal2):
    '''
    Gets signal of maximum intensity

    signal1: signal from channel 1 (array-like)
    signal2: signal from channel 2 (array-like)

    returns:
    sig: maximum between the two signals (array-like)
    '''
    if max(abs(signal1)) > max(abs(signal2)):
        sig=signal1
    else:
        sig=signal2
    return sig
