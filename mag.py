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
import find as find
import scipy.signal as sig

################# bring in data ###################

df = pd.read_csv('/Users/kierapond/Documents/GitHub/multi-mag-thesis/trial1.csv',skiprows=range(0,17),low_memory=False)

#print(df)

mag3column = ['CH1']

#print(mag3column)

mag3 = df[mag3column].to_numpy()

#print(mag3)

#mag3factor = [i * 10 for i in mag3]

#print(mag3factor)

mag4column = ['CH2']

mag4 = df[mag4column].to_numpy()

trigcolumn = ['CH4']

trig = df[trigcolumn].to_numpy()

timecolumn = ['TIME']

time = df[timecolumn].to_numpy()

#print(time)

# plt.plot(time,mag4)
# plt.plot(time,mag3)
# plt.xlim(0, 100000)
# plt.show()

# plt.plot(trig)
# plt.xlim(0, 100000)
# plt.show()

################# find data #########################

pumpdata = trig


# 1 - find "derivative" - differences btw points
# 2 - find LARGE differences btw points to find spikes 
# 3 - get rid of multiple peaks v close together to get just one number
    #for each spike 
# create starts and ends   
    

# pump is trig and probe is mag signal 


inds = find.find_ind(df['CH4'])
starts = inds[0,:]
ends = inds[1,:]

# print(starts)
# print(ends)

# print(mag3[starts[1]])
# print(mag3[ends[1]])

fittime = time[starts[1]:ends[1]]
# print(fittime)
# print(type(fittime))

# fitmag3 = mag3[starts[1]:ends[1]]
# print(fitmag3)

n=8

# plt.plot(time, mag3)
# plt.show()
# plt.figure(2)
plt.plot(time[starts[n]:ends[n]],mag3[starts[n]:ends[n]])
plt.show()

tapsraw = open('taps_for_kiera.txt' , 'r', newline = '\n').readlines()
taps = [float(t) for t in tapsraw]
# print(np.shape(mag3))
# print(np.shape(taps))
# print(type(mag3))
# print(type(taps))

# mag3flat = mag3.flatten()
# print(mag3flat)

probedata = sig.oaconvolve(mag3[:,0], taps, mode='same')


plt.figure(2)
plt.plot(time[starts[n]:ends[n]],probedata[starts[n]:ends[n]])
plt.show()

fitdata = probedata[starts[n]:ends[n]]

time = time[starts[n]:ends[n]] 

fittime = time - min(time)

plt.figure(3)
plt.plot(fittime,fitdata)
plt.show()

# ends = np.delete(ends, 0)
# ends = np.append(ends, len(df['CH4']))



################# fit data #########################


def fitfun(x,a,b,c,d,f):
    return a*np.exp(-b*(x))*np.sin(c*(x)+d)+f

for i in range(0,len(mag3[starts[n]])):
    ampguess = abs(max(fitdata) - min(fitdata))/2
    tconstguess = 50
    wguess = 7000
    phaseguess = np.pi/4
    offsetguess = np.mean(fitdata)
    guess = [ampguess,tconstguess,wguess,phaseguess,offsetguess]
    bounds = [[0,0,0,-2*np.pi,-ampguess], 
             [10,200,2*wguess,2*np.pi,ampguess]]
    popt, pcov = scipy.optimize.curve_fit(fitfun, fittime, fitdata, 
            bounds = bounds, p0 = guess, maxfev = 50000, ftol = 1e-15,
            gtol = 1e-15, xtol = 1e-15)
    
    if i == 0:
        params = popt
        perr = np.sqrt(np.diag(pcov))
        
    else: 
        params = np.vstack((params, popt))
        perr = np.vstack((perr, np.sqrt(np.diag(pcov))))
        

print(params, perr)

    




# def fit_FIDS(metadata):
#     for i in range(0,len(metadata['starts'])):
#         ydata = df['Probe']
        
        
        
        
        

################# show data #########################






