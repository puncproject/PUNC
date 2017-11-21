# Imports important python 3 behaviour to ensure correct operation and
# performance in python 2
from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

from dolfin import *
from punc import *
from numpy import pi
import numpy as np
from matplotlib import pyplot as plt
import time

#==============================================================================
# INITIALIZING FENICS
#------------------------------------------------------------------------------

uniform = False
periodic = True
nDims = 1                           # Number of dimensions

Ld = 1*np.ones(nDims)               # Length of domain
Nr = 64*np.ones(nDims,dtype=int)    # Number of 'rectangles' in mesh

if uniform:
    if nDims == 1:
        mesh = IntervalMesh(Nr[0],0,Ld[0])
    elif nDims == 2:
        mesh = RectangleMesh(Point(0,0),Point(Ld),*Nr)
    elif nDims == 3:
        mesh = BoxMesh(Point(0,0,0),Point(Ld),*Nr)
else:
    if nDims == 1:
        mesh = Mesh("mesh/nonuniform_interval.xml")
    elif nDims == 2:
        mesh = Mesh("mesh/nonuniform.xml")



constr = PeriodicBoundary(Ld) if periodic else None
V = FunctionSpace(mesh, 'CG', 1, constrained_domain=constr)

print("Running voronoi_length:")
t = time.time()
lengths = voronoi_length(V, Ld, periodic, raw=False)
print(time.time()-t,"s")

print("Running Voro++")
t = time.time()
volumes = voronoi_volume(V, Ld, periodic, raw=True)
print(time.time()-t,"s")

print("Max difference:")
print(max(abs(volumes-lengths.vector().get_local())))

plot(lengths)
interactive()
