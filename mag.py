#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:58:15 2022

@author: kierapond
"""

import pandas as pd
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

################# bring in data ###################

df = pd.read_csv('/Users/kierapond/Documents/GitHub/multi-mag-thesis/trial1.csv',skiprows=range(0,17),low_memory=False)

print(df)

#mag3 = CH1

#mag4 = CH2

#trigger = CH3


################# find data #########################


def find_ind(pumpdata):
    
    diffs = np.diff(pumpdata) #1st order DE
    
    plt.figure(1) #? what is this for ? 
    plt.plot(pumpdata)
    thresh = np.abs(0.3*(max(diffs) - min(diffs)))
    
    plt.figure(2) #? what is this for ? 
    plt.plot(diffs)
    plt.show()
    
    starts = np.where(diffs < -1*thresh)[0] #negative spike  
    ends = np.where(diffs > thresh)[0] #positive spike
    startDiff = np.diff(starts)
    endDiff = np.diff(ends)
    #print(starts)
    #print(ends)
    startDel = np.where(startDiff < 100)[0] #double sampled peaks
    endDel = np.where(endDiff < 100)[0]
    
    newstarts = np.delete(starts, startDel) # deleting double peakrs
    newends = np.delete(ends, endDel)
    
    if len(newstarts) != len(newends):
        if newstarts[0] < newends[0]:
            newstarts = np.delete(newstarts, len(newstarts))
        else: 
            newends = np.delete(newends, len(newends))
    elif newends[0] < newstarts[0]:
        newstarts = np.delete(newstarts, len(newstarts))
        newends = np.delete(newends,0) 
        
    return np.stack((newstarts, newends), axis=0)

inds = find_ind(df['CH3'])
starts = inds[0,:]
ends = inds[1,:]
ends = np.delete(ends, 0)
ends = np.append(ends, len(df['CH3']))



################# fit data #########################


def fitfun(x,a,b,c,d,f):
    return a*np.exp(-b*(x))*np.sin(c*(x)+d)+f



def fit_FIDS(metadata):
    for i in range(0,len(metadata['starts'])):
        ydata = df['Probe']
        
        
        
        
        

################# show data #########################






