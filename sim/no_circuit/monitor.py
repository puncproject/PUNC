#!/usr/bin/env python

# Live monitoring of simulation results. Run simulations, and run
# ./monitor.py live
#

from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import numpy as np
import matplotlib.pyplot as plt
import time
import sys

live = True if 'live' in sys.argv else False;
save = True if 'save' in sys.argv else False;
show = True if 'show' in sys.argv else False;

if live and show:
    raise ValueError("'live' and 'show' cannot be both on.")

if live and save:
    raise ValueError("'live' and 'save' cannot be both on.")

if live: plt.ion()

defweight = 0.02
dpi = 300

def expAverage(data, weight=defweight):
    result = np.zeros(data.shape)
    result[0] = data[0]
    for i in range(1,len(data)):
        result[i] = weight*data[i] + (1-weight)*result[i-1]
    return result

while(True):
    f = open('history.dat')
    data = np.array([l.split('\t') for l in f], dtype=float)
    f.close()

    fig = plt.figure(1)
    fig.clear()
    plt.plot(data[:,0], data[:,2],label='electrons')
    plt.plot(data[:,0], data[:,3],label='ions')
    # plt.plot(data[:,0], data[:,1]+data[:,2],':',label='total')
    plt.legend(loc="lower left")
    plt.xlabel("Timestep")
    plt.ylabel("Number")
    plt.title("Number of particles")
    plt.grid()
    if save: plt.savefig('particles.png', format='png', dpi=dpi)

    fig = plt.figure(2)
    fig.clear()
    plt.plot(data[:,0], data[:,4],label='kinetic')
    plt.plot(data[:,0], data[:,5],label='potential')
    plt.plot(data[:,0], data[:,4]+data[:,5],':',label='total')
    plt.legend(loc="lower left")
    plt.xlabel("Timestep")
    plt.ylabel("PUNC energy units")
    plt.title("Energy")
    plt.grid()
    if save: plt.savefig('energy.png', format='png', dpi=dpi)

    fig = plt.figure(3)
    fig.clear()
    plt.plot(data[:,0], data[:,7], '#999999', label='raw data')
    plt.plot(data[:,0], expAverage(data[:,7]), label='exp. average (%.2f)'%defweight)
    plt.legend(loc="lower left")
    plt.xlabel("Timestep")
    plt.ylabel("Laframboise units")
    plt.title("Object potential")
    plt.grid()
    if save: plt.savefig('potential.png', format='png', dpi=dpi)

    fig = plt.figure(4)
    fig.clear()
    plt.plot(data[:,0], data[:,8], '#999999', label='raw data')
    plt.plot(data[:,0], expAverage(data[:,8]), label='exp. average (%.2f)'%defweight)
    plt.legend(loc="lower left")
    plt.xlabel("Timestep")
    plt.ylabel("Laframboise units")
    plt.title("Current collected by object")
    plt.grid()
    if save: plt.savefig('current.png', format='png', dpi=dpi)

    if live: plt.draw()
    if show: plt.show()

    # Emits warning. Seems like this is an unsolved problem in matplotlib.
    plt.pause(1)

    if not live: break
