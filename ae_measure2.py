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
    csv = pandas.read_csv(filter_csv)
    events = np.array(csv.Event)
    ev = events[np.where(np.isnan(events) == False)]
    ev = ev.astype(int)
    v1,v2 = read_ae_file2(ae_file)
    v1 = np.array(v1); v2 = np.array(v2)
    v1 = v1[ev-1]
    v2 = v2[ev-1]
    return v1,v2,ev

def read_ae_file(fname):
    f = open(fname)
    lines = f.readlines()[1:]
    v1 = np.array([
        float(line.split()[0]) for line in lines])
    v2 = np.array([
        float(line.split()[1]) for line in lines])
    f.close()
    return v1,v2

'''
Outputs an array v1, v2 whose elements are an individual signal (set of 1024 data points)
'''
def read_ae_file2(fname):
    f = open(fname)
    lines = f.readlines()[1:]
    v1 = np.array([
        float(line.split()[0]) for line in lines])
    v2 = np.array([
        float(line.split()[1]) for line in lines])
    f.close()
    v1s = []
    v2s = []
    for i in range(0,len(v1),1024):
        v1s.append(v1[i:i+1024])
        v2s.append(v2[i:i+1024])
    return v1s,v2s

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
    #dt = t[1]-t[0]
    z = np.abs(np.fft.fft(y))
    w = np.arange(len(z))
    w = (w/dt)/(len(z)//2)
    return z,w

sos300 = signal.butter(10, 300e3, 'hp', fs=20000000, output='sos')

def filter300(sig):
    filtered = signal.sosfilt(sos300,sig)
    return filtered
