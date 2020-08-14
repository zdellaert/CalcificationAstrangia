import numpy as np
import skimage.feature as sf

def threshold(a,t):
    return np.percentile(a, t) 


def balance(a,t1=0.7, t2=0.3):
    
    m = a.mean()
    p1 = np.percentile(a, t1) 
    p2 = np.percentile(a, t2) 

    return (p1 - m) / (m - p2)


def skewness(a):

    N = a.size
    i, n = np.unique(a,  return_counts=True)
    i_bar = np.sum(i*n)/N

    m2 = np.sum(n*(i- i_bar)**2)/N
    m3 = np.sum(n*(i- i_bar)**3)/N

    return m3/(m2**1.5)


def energy(a):
    i, n = np.unique(a,  return_counts=True)
    p = n/float(a.size)
    
    return np.sum(p**2)
    

def entropy(a):
    i, n = np.unique(a,  return_counts=True)
    p = n/float(a.size)
    
    return -np.sum(p*np.log(p))


def contrast(a,d=None,ang=None,l=256):

    #d: distance
    #ang: angle

    if d==None:
        d = [1]

    if ang==None:
        ang = np.linspace(0,2*np.pi,100)

    g = sf.greycomatrix(a.astype(np.uint8), [1], ang, levels=l,normed=True, symmetric=True)

    return np.sum(sf.greycoprops(g, 'contrast'))


def mean_gradient(a, d):
    return np.mean(np.gradient(a))


def RMS(a):
    
    A = np.fft.fft2(a)
    
    return np.sqrt(np.sum(np.abs(A)**2))


def FMP(a, d=1):
    
    A = np.abs(np.fft.fft2(a))**2

    u1 = np.fft.fftfreq(A.shape[0],d=d)
    v1 = np.fft.fftfreq(A.shape[1],d=d)

    v2, u2 = np.meshgrid(v1,u1)

    return np.sum(np.sqrt(u2**2 + v2**2) * A) / np.sum(A)
    


import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
