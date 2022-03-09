#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:58:15 2022

@author: kierapond
"""


import numpy as np
import matplotlib.pyplot as plt



def find_ind(pumpdata):
    
    diffs = np.diff(pumpdata) #1st order DE
    # print(diffs)
    
    #plt.plot(pumpdata)
    thresh = np.abs(0.1*(max(diffs) - min(diffs)))
    
    # plt.plot(diffs)
    # plt.show()
    
    starts = np.where(diffs < -1*thresh)[0] #negative spike  
    ends = np.where(diffs > thresh)[0] #positive spike
    startDiff = np.diff(starts)
    endDiff = np.diff(ends)
    # print(starts)
    # print(ends)
    startDel = np.where(startDiff < 100)[0] #double sampled peaks
    endDel = np.where(endDiff < 100)[0]
    
    newstarts = np.delete(starts, startDel) # deleting double peakrs
    newends = np.delete(ends, endDel)
    # print(newstarts)
    # print(newends)
    # print(len(newstarts))
    # print(len(newends))
    # print(newstarts[0])
    # print(newends[0])
    # print(thresh)
    
    # print(newstarts[len(newstarts)-1])
    # plt.plot(newstarts)
    # plt.plot((newends))
    
    if len(newstarts) != len(newends):
        if newstarts[0] < newends[0]:
            newstarts = np.delete(newstarts, len(newstarts))
        else: 
            newends = np.delete(newends, len(newends))
    elif newends[0] < newstarts[0]:
        newstarts = np.delete(newstarts, len(newstarts)-1)
        newends = np.delete(newends,0)
        
    # print(len(newstarts))
    # print(len(newends))
    # print(newstarts)
    # print(newends)
    
    return np.stack((newstarts, newends), axis=0)
