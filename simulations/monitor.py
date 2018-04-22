#!/usr/bin/env python

# Live monitoring of simulation results. Run simulations, and run
# ./monitor.py live
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

def expAvg(data, dt=0.002, tau=0.1, weight=None):
    """
    Makes an exponential moving average of "data". dt is the timestep between
    each sample in some unit and tau is the relaxation time in the same unit.
    """
    weight = 1-np.exp(-dt/tau)
    result = np.zeros(data.shape)
    result[0] = data[0]
    for i in range(1,len(data)):
        result[i] = weight*data[i] + (1-weight)*result[i-1]
    return result

def plotAvg(x, y, label=None, tau=0.1, linewidth=1):
    """
    Plots a moving exponential average of "y" versus "x" in a matplotlib plot
    while showing the raw values of "y" in the background. tau is the relaxation
    time in the same unit as the value on the x-axis.
    """
    dx = x[1]-x[0]
    plt.plot(x, y, '#CCCCCC', linewidth=1, zorder=0)
    p = plt.plot(x, expAvg(y, dx, tau), linewidth=1, label=label)
    return p

fig1 = plt.figure(1)
ax = fig1.gca()
x1, y1 = [], []
ln, = plt.plot(x1, y1, linewidth=1, animated=True)
# ax.set_autoscaley_on(True)
# ax.set_autoscalex_on(True)

def init():
    return ln,

def update(frame):
    f = open('history.dat')
    data = np.array([l.split('\t') for l in f], dtype=float)
    f.close()

    x1 = data[:,0]
    y1 = data[:,6]
    ln.set_data(x1,y1)
    ax.relim()
    ax.autoscale_view()
    # ax.set_xlim(min(x1),max(x1))
    # ax.set_ylim(min(y1),max(y1))
    # ax.autoscale()
    return ln,

ani = FuncAnimation(fig1, update, init_func=init, blit=False)

plt.show()

# while(True):
 #    f = open('history.dat')
 #    data = np.array([l.split('\t') for l in f], dtype=float)
 #    f.close()

 #    xaxis = data[:,0]

 #    fig = plt.figure(1)
 #    fig.clear()
 #    plt.plot(xaxis, data[:,1], label='electrons', linewidth=1)
 #    plt.plot(xaxis, data[:,2], label='ions', linewidth=1)
 #    # plt.plot(xaxis, data[:,1]+data[:,2],':', label='total', linewidth=1)
 #    plt.legend(loc="lower left")
 #    plt.xlabel("Timestep")
 #    plt.ylabel("Number")
 #    plt.title("Number of particles")
 #    plt.grid()
 #    if save: plt.savefig('particles.png', format='png', dpi=dpi)

 #    fig = plt.figure(2)
 #    fig.clear()
 #    plt.plot(xaxis, data[:,3], label='kinetic', linewidth=1)
 #    plt.plot(xaxis, data[:,4], label='potential', linewidth=1)
 #    plt.plot(xaxis, data[:,3]+data[:,4],':', label='total', linewidth=1)
 #    plt.legend(loc="lower left")
 #    plt.xlabel("Timestep")
 #    plt.ylabel("PUNC energy units")
 #    plt.title("Energy")
 #    plt.grid()
 #    if save: plt.savefig('energy.png', format='png', dpi=dpi)

 #    fig = plt.figure(3)
 #    fig.clear()
 #    plotAvg(xaxis, data[:,5])
 #    plt.xlabel("Timestep")
 #    plt.ylabel("Laframboise units")
 #    plt.title("Object potential")
 #    plt.grid()
 #    if save: plt.savefig('potential.png', format='png', dpi=dpi)

 #    fig = plt.figure(4)
 #    fig.clear()
 #    plotAvg(xaxis, data[:,6], tau=40)
 #    plt.xlabel("Timestep")
 #    plt.ylabel("Laframboise units")
 #    plt.title("Current collected by object")
 #    plt.grid()
 #    if save: plt.savefig('current.png', format='png', dpi=dpi)

 #    if live: plt.draw()
 #    if show: plt.show()

 #    # Emits warning. Seems like this is an unsolved problem in matplotlib.
 #    plt.pause(1)

 #    if not live: break
