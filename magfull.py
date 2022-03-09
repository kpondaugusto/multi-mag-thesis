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

#print(df)

mag3column = ['CH1']

#print(mag3column)

mag3 = df[mag3column].to_numpy()

# print(mag3)

mag4column = ['CH2']

mag4 = df[mag4column].to_numpy()

trigcolumn = ['CH4']

trig = df[trigcolumn].to_numpy()

timecolumn = ['TIME']

time = df[timecolumn].to_numpy()

time = range(0,10000000,1)

#print(time)

plt.plot(time,mag4)
plt.plot(time,mag3)
plt.xlim(0, 100000)
plt.show()

plt.plot(trig)
plt.xlim(0, 100000)
plt.show()

################# find data #########################


# 1 - find "derivative" - differences btw points
# 2 - find LARGE differences btw points to find spikes 
# 3 - get rid of multiple peaks v close together to get just one number
    #for each spike 
# create starts and ends   
    

# pump is trig and probe is mag signal 


def find_ind(pumpdata):
    
    diffs = np.diff(pumpdata) #1st order DE
    print(diffs)
    
    # plt.plot(pumpdata)
    thresh = np.abs(0.3*(max(diffs) - min(diffs)))
    
    plt.plot(diffs)
    plt.show()
    
    starts = np.where(diffs < -1*thresh)[0] #negative spike  
    ends = np.where(diffs > thresh)[0] #positive spike
    startDiff = np.diff(starts)
    endDiff = np.diff(ends)
    # print(endDiff)
    # print(startDiff)
    # print(starts)
    # print(ends)
    startDel = np.where(startDiff < 100)[0] #double sampled peaks
    endDel = np.where(endDiff < 100)[0]
    # print(endDel)
    # print(startDel)
    
    newstarts = np.delete(starts, startDel) # deleting double peakrs
    newends = np.delete(ends, endDel)
    print(newstarts)
    print(newends)
    print(len(newstarts))
    print(len(newends))  
    print(newstarts[0])
    print(newends[0])
    
    if len(newstarts) != len(newends):
        if newstarts[0] < newends[0]:
            newstarts = np.delete(newstarts, len(newstarts))
        else: 
            newends = np.delete(newends, len(newends)-1)
    elif newends[0] < newstarts[0]:
        newstarts = np.delete(newstarts, len(newstarts))
        newends = np.delete(newends,0) 
        
    print(len(newstarts))
    print(len(newends))    
    return np.stack((newstarts, newends), axis=0)


inds = find_ind(df['CH4'])
# starts = inds[0,:]
# ends = inds[1,:]
# ends = np.delete(ends, 0)
# ends = np.append(ends, len(df['CH4']))



################# fit data #########################


def fitfun(x,a,b,c,d,f):
    return a*np.exp(-b*(x))*np.sin(c*(x)+d)+f



# def fit_FIDS(metadata):
#     for i in range(0,len(metadata['starts'])):
#         ydata = df['Probe']
        
        
        
        
        

################# show data #########################






