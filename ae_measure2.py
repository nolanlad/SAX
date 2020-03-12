'''
Author: Nolan McCarthy
Contact: nolanrmccarthy@gmail.com
Version: 200312

This is a class definition and function definition file for parsing AE signals
as developed by the Daly Lab
'''

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pandas
from scipy.stats import linregress


nan = float('nan')
inf = float('inf')

'''
We should talk about the other functions in this file, i.e. everything that isn't
read_ae_file or read_ae_file2. They may become nessecary later, however we might
want to consider moving things around for readability.
'''

def filter_ae(ae_file, filter_csv):
    '''
    ae_file: text file containing voltage over time
    filter_csv: file of useful events, must be csv
    return: (v1, v2, event_num), where v1 and v2 are the within gauge signals and
            event_number is the event number indexed from 1
    '''
    csv = pandas.read_csv(filter_csv)
    events = np.array(csv.Event)
    ev = events[np.where(np.isnan(events) == False)]
    ev = ev.astype(int)
    v1,v2, _ = read_ae_file2(ae_file)
    v1 = np.array(v1); v2 = np.array(v2)
    v1 = v1[ev-1]
    v2 = v2[ev-1]
    return v1, v2, ev



def is_clipped(sig):
    '''
    Determines if a signal is clipped
    sig: AE signal (N*1 array-like)
    return: bool (boolean)
    '''
    bool = False
    sig = np.abs(sig)
    max = np.max(np.round(sig, decimals=2))
    sig = sig[np.where(sig==max)]
    if len(sig)>1:
        bool=True
    return bool

def remove_clipped(v1, v2, ev, time = []):
    '''
    if both signals are clipped, removes both signals.
    v1: Signal from channel 1 (N*1 array-like)
    v2: Signal from channel 2 (N*1 array-like)
    event_num: where v1 and v2 are the within gauge signals and
            event_number is the event number indexed from 1
    return: (v1, v2, event_num), where v1 and v2 are the within gauge signals and
            event_number is the event number indexed from 1
    '''
    if len(time)!=0:
        holder1 = []
        holder2 = []
        holder3 = []
        holder4 = []
        for i in range(len(v1)):
            if is_clipped(v1[i]) and is_clipped(v2[i]):
                pass
            else:
                holder1.append(v1[i])
                holder2.append(v2[i])
                holder3.append(ev[i])
                holder4.append(time[i])
        return holder1, holder2, holder3, holder4
    else:
        holder1 = []
        holder2 = []
        holder3 = []
        for i in range(len(v1)):
            if is_clipped(v1[i]) and is_clipped(v2[i]):
                pass
            else:
                holder1.append(v1[i])
                holder2.append(v2[i])
                holder3.append(ev[i])
                holder4.append(time[i])
        return holder1, holder2, holder3




def read_ae_file2(fname, sig_length=1024):
    '''
    fname: text file containing voltage over time (string)
    sig_length: number of data points per signal (int)

    returns:
    (v1, v2, ev): where v1 and v2 are signals and
            ev is the event number indexed from 1
    '''
    f = open(fname)
    lines = f.readlines()[1:]
    v1 = np.array([
        float(line.split()[0]) for line in lines])
    v2 = np.array([
        float(line.split()[1]) for line in lines])
    f.close()

    v1s = []
    v2s = []
    for i in range(0,len(v1),sig_length):
        v1s.append(v1[i:i+sig_length])
        v2s.append(v2[i:i+sig_length])
    ev = list(range(int(len(v1)/sig_length))) # makes list [0,1,2,...,n]
    ev = np.array([x+1 for x in ev]) #indexes from 1 and casts as list
    v1s = np.array(v1s)
    v2s = np.array(v2s)

    return v1s, v2s, ev



def max_sig(signal1, signal2):
    '''
    Gets signal of maximum intensity, currrently

    signal1: signal from channel 1, single event (array-like)
    signal2: signal from channel 2, single event (array-like)

    returns:
    sig: maximum between the two signals (array-like)
    '''
    if max(abs(signal1)) > max(abs(signal2)):
        sig=signal1
    else:
        sig=signal2
    return sig


def min_sig(signal1, signal2):
    '''
    Gets signal of maximum intensity, currrently

    signal1: signal from channel 1, single event (array-like)
    signal2: signal from channel 2, single event (array-like)

    returns:
    sig: maximum between the two signals (array-like)
    '''
    if max(abs(signal1)) > max(abs(signal2)):
        sig=signal2
    else:
        sig=signal1
    return sig







def read_ae_file(fname):
    f = open(fname)
    lines = f.readlines()[1:]
    v1 = np.array([
        float(line.split()[0]) for line in lines])
    v2 = np.array([
        float(line.split()[1]) for line in lines])
    f.close()
    return v1,v2


def get_first_peak(sig,thresh = 1.2):
    trig = np.amax(np.abs(sig[:150]))*thresh
    peaks = signal.find_peaks(sig,height=trig)
    if len(peaks[0]) == 0:
        return nan
    return peaks[0][0]

def get_first_peak2(sig,thresh = 0.2):
    trig = (np.amax((sig[:]))-np.amax((sig[:150])))*thresh
    peaks = signal.find_peaks(sig,height=trig)
    if peaks == []:
        return nan
    return peaks[0][0]

def find_dt(sig1,sig2,thresh=1.2):
    at1 = get_first_peak(sig1,thresh=thresh)*0.1
    at2 = get_first_peak(sig2,thresh=thresh)*0.1
    dt = at2 - at1
    return dt

def get_first_peak3(sig,thresh = 1.2):
    sig = noise_reduce(sig)
    trig = np.amax(np.abs(sig[:150]))*thresh
    peaks = signal.find_peaks(sig,height=trig)
    return peaks[0][0]

def noise_reduce(sig):
    noise = sig[:150]
    z = np.abs(np.fft.fft(noise))
    peaks = signal.find_peaks(z,height = 0.02)[0]
    b,a = signal.iirfilter(2,peaks/150)
    return signal.lfilter(b,a,sig)

def good_fft(dt,y):
    z = np.abs(np.fft.fft(y))
    w = np.arange(len(z))
    w = (w/dt)/(len(z)//2)
    return z,w
