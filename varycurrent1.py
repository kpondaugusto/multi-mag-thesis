#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 16:01:20 2022

@author: kierapond
"""


import pandas as pd
import numpy as np
import scipy.optimize
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import scipy.signal as sig
import allantools as a

################# bring in data ###################

df = pd.read_csv('/Users/kierapond/Documents/GitHub/multi-mag-thesis/vary1.csv',skiprows=range(0,17),low_memory=False)

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


################# find data #########################

pumpdata = trig


# 1 - find "derivative" - differences btw points
# 2 - find LARGE differences btw points to find spikes 
# 3 - get rid of multiple peaks v close together to get just one number
    #for each spike 
# create starts and ends   
    

# pump is trig and probe is mag signal 

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
            newstarts = np.delete(newstarts, len(newstarts)-1)
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


inds = find_ind(df['CH4'])
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
# plt.plot(time[starts[n]:ends[n]],mag3[starts[n]:ends[n]])
# plt.show()

tapsraw = open('taps_for_kiera.txt' , 'r', newline = '\n').readlines()
taps = [float(t) for t in tapsraw]
# print(np.shape(mag3))
# print(np.shape(taps))
# print(type(mag3))
# print(type(taps))

# mag3flat = mag3.flatten()
# print(mag3flat)

probedata = sig.oaconvolve(mag3[:,0], taps, mode='same')

probedatamag4 = sig.oaconvolve(mag4[:,0], taps, mode='same')

# plt.figure(2)
# plt.plot(time[starts[n]:ends[n]],probedata[starts[n]:ends[n]])
# plt.show()

fitdata1 = probedata[starts[n]:ends[n]]

fitdatamag4 = probedatamag4[starts[n]:ends[n]]

time1 = time[starts[n]:ends[n]] 

fittime1 = time - min(time)


################# fit data #########################


def fitfun(x,a,b,c,d,f):
    return a*np.exp(-b*(x))*np.sin(c*(x)+d)+f

for i in range(0,len(mag3[starts])):
    fitdata = probedata[starts[i]:ends[i]]
    time2 = time[starts[i]:ends[i]] 
    fittime = time2 - min(time2)
    ampguess = abs(max(fitdata) - min(fitdata))/2
    tconstguess = 50
    wguess = 2*np.pi*7000
    phaseguess = np.pi/4
    offsetguess = np.mean(fitdata)
    guess = [ampguess,tconstguess,wguess,phaseguess,offsetguess]
    bounds = [[0,0,0,-2*np.pi,-ampguess], 
             [10,200,2*wguess,2*np.pi,ampguess]]
    popt, pcov = scipy.optimize.curve_fit(fitfun, fittime[:,0], fitdata, 
            p0 = guess, bounds = bounds, maxfev = 50000, ftol = 1e-15, 
            gtol = 1e-15, xtol = 1e-15)
    
    if i == 0:
        params = popt
        perr = np.sqrt(np.diag(pcov))
        
    else: 
        params = np.vstack((params, popt))
        perr = np.vstack((perr, np.sqrt(np.diag(pcov))))
        
        
for i in range(0,len(mag4[starts])):
    fitdatamag4 = probedatamag4[starts[i]:ends[i]]
    timemag4 = time[starts[i]:ends[i]] 
    fittimemag4 = timemag4 - min(timemag4)
    ampguessmag4 = abs(max(fitdatamag4) - min(fitdatamag4))/2
    tconstguessmag4 = 100
    wguessmag4 = 2*np.pi*7000
    phaseguessmag4 = np.pi/4
    offsetguessmag4 = np.mean(fitdatamag4)
    guessmag4 = [ampguessmag4,tconstguessmag4,wguessmag4,phaseguessmag4,offsetguessmag4]
    boundsmag4 = [[0,0,0,-2*np.pi,-2*ampguessmag4], 
             [100,2000,2*wguessmag4,2*np.pi,2*ampguessmag4]]
    poptmag4, pcovmag4 = scipy.optimize.curve_fit(fitfun, fittimemag4[:,0], fitdatamag4, 
            p0 = guessmag4, bounds = boundsmag4, maxfev = 50000, ftol = 1e-15, 
            gtol = 1e-15, xtol = 1e-15)
    
    if i == 0:
        paramsmag4 = poptmag4
        perrmag4 = np.sqrt(np.diag(pcovmag4))
        
    else: 
        paramsmag4 = np.vstack((paramsmag4, poptmag4))
        perrmag4 = np.vstack((perrmag4, np.sqrt(np.diag(pcovmag4))))
                

print(params)

        
diff34 = params[:,2]/(2*np.pi*7.1096) - paramsmag4[:,2]/(2*np.pi*7.1096) #difference btw mag3&4

################ allan deviation #########################

t = np.logspace(-1, 3)

(t2,ad,ade,adn)= a.oadev(params[:,2]/(2*np.pi*7.1096),rate=10,data_type='freq',taus=t)

(t2mag4,admag4,ademag4,adnmag4)= a.oadev(paramsmag4[:,2]/(2*np.pi*7.1096),rate=10,data_type='freq',taus=t)

(t234,ad34,ade34,adn34)= a.oadev(diff34,rate=10,data_type='freq',taus=t)


################ show data #########################


plt.plot(fittime,fitdata)
plt.plot(fittime,fitfun(fittime,*params[i,:]))
# plt.plot(fittime,fitfun(fittime,0.05,10,2*np.pi*7000,0,0))
plt.xlabel('Time (s)')
plt.ylabel('Signal Amplitude (V)')
plt.show()

plt.figure(3)
plt.plot(fittimemag4,fitdatamag4)
plt.plot(fittimemag4,fitfun(fittimemag4,*paramsmag4[i,:]))
# plt.plot(fittime,fitfun(fittime,0.05,10,2*np.pi*7000,0,0))
plt.xlabel('Time (s)')
plt.ylabel('Signal Amplitude (V)')
plt.show()


plt.figure(2)
plt.plot(params[:,2]/(2*np.pi*7.1096),label='Sensor 3')
plt.plot(paramsmag4[:,2]/(2*np.pi*7.1096), label='Sensor 4')
plt.legend()
plt.ylabel('Field (nT)')
plt.xlabel('Number of FID')
plt.show()


plt.figure(4)
plt.plot(diff34)
plt.ylabel('Difference Between Sensor 3 and 4 (nT)') #in caption it is proportional to field
plt.xlabel('Number of FID')
plt.show()

# plt.figure(5)
# plt.plot(t2,ad,label='Sensor 3')
# plt.plot(t2mag4,admag4,label='Sensor 4')
# plt.plot(t234,ad34,label='Difference Btw 3&4')
# plt.yscale("log")
# plt.xscale("log")
# plt.ylabel('Time (s)')
# plt.xlabel('Log(Frequency)')
# plt.legend()
# plt.show()





