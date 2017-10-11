from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import time as t
import numpy as np
import sys
from collections import OrderedDict

class Timer(object):
    """
    Acts like a stop watch. Start with start() and stop with stop(). stop() also
    acts like a lap key, e.g. it can be called several times. The Timer will
    compute a running mean and standard deviation of lap times. Storage is O(1).
    Call stop(True) to get a verbose output, or read the variables manually.
    reset() will reset the timer. Example:

        t = Timer()

        t.start()
        # lap 1
        t.stop()
        # lap 2
        t.stop()

        # does not count

        t.start()
        # lap 3
        t.stop()

    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0
        self.laps = 0
        self.last = 0
        self.mean = 0
        self.mean_squared = 0
        self.stdev = 0

    def start(self):
        self._start = t.time()

    def stop(self, verbose=False):
        new_time = t.time()
        elapsed = new_time - self._start
        self._start = new_time

        self.laps += 1
        self.last = elapsed
        self.total += elapsed

        self.mean += (elapsed - self.mean)/self.laps
        self.mean_squared += (elapsed**2 - self.mean_squared)/self.laps

        var = self.laps/max((self.laps-1),1)*(self.mean_squared-self.mean**2)
        self.stdev = np.sqrt(var)

        if(verbose): print(self)

    def __str__(self):
        s  = "last : %f\n"%self.last
        s += "total: %f\n"%self.total
        s += "mean : %f\n"%self.mean
        s += "stdev: %f\n"%self.stdev
        s += "laps : %d"%self.laps
        return s

class TaskTimer(object):
    """
    Tracks timing of several tasks in a loop and maintains progress status and
    statistics. Example:

    """

    def __init__(self, laps, mode='compact'):
        assert mode in ['simple','compact','quiet']
        self.mode = mode

        self.master = Timer()
        self.timers = OrderedDict()
        self.current = None
        self.laps = laps

    def task(self, tag):

        if self.current == None:
            self.master.start()

        if self.current != None:
            self.timers[self.current].stop()

        if not tag in self.timers:
            self.timers[tag] = Timer()

        self.timers[tag].start()
        self.current = tag

        if(self.mode=='compact'):
            print("\r",self,end='',sep='')
            sys.stdout.flush()

    def end(self):

        if self.current != None:
            self.timers[self.current].stop()
            self.master.stop()
            self.current = None

        if(self.mode=='simple'):
            print(self)

        if(self.mode=='compact'):
            print("\r",self,end='',sep='')
            sys.stdout.flush()
            if(self.master.laps==self.laps): print("\n")

    def summary(self):

        row = ['','Mean','StDev','Total','%']
        table = [row]

        K = list(self.timers.keys())
        V = list(self.timers.values())
        K.append('Total')
        V.append(self.master)

        for k,v in zip(K,V):
            fraction = 100
            if self.master.total!=0:
                fraction = 100*v.total/self.master.total
            row = [k, format_time(v.mean),
                      format_time(v.stdev),
                      format_time(v.total),
                      '{:.0f}'.format(fraction)]
            table.append(row)

        self.table = table

        colwidth = np.max(np.array([[len(a) for a in b] for b in table]),0)
        colwidth[1:] += 2 # column spacing

        s = ''
        for row in table:
            s += '{:<{w}}'.format(row[0],w=colwidth[0])
            for col,w in zip(row[1:],colwidth[1:]):
                s += '{:>{w}}'.format(col,w=w)
            s += "\n"

        print(s)

    def __str__(self):

        lap = self.master.laps
        total_time = (self.laps/max(lap,1))*self.master.total
        eta = total_time-self.master.total
        progress = 100.0*lap/self.laps
        width = max(len(a) for a in self.timers.keys())
        current = '' if self.current==None else self.current

        s  = "Completed step %i/%i (%.0f%%). "%(lap,self.laps,progress)
        s += "ETA: {} (total: {}). ".format(format_time(eta),format_time(total_time))
        s += "{:<{w}}".format(current,w=width)
        return s

def format_time(seconds):
    s = int(seconds % 60)
    m = int(seconds / 60)
    if(m>0):
        return "{:d}:{:02d}".format(m,s)
    else:
        return "{:d}s".format(s)
