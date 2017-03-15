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

#==============================================================================
# INITIALIZING FENICS
#------------------------------------------------------------------------------

nDims = 1                           # Number of dimensions
Ld = 1*np.ones(nDims)            # Length of domain
Nr = 32*np.ones(nDims,dtype=int)    # Number of 'rectangles' in mesh

if nDims == 1:
    mesh = IntervalMesh(Nr[0],0,Ld[0])
elif nDims == 2:
    mesh = RectangleMesh(Point(0,0),Point(Ld),*Nr)

# mesh = Mesh("mesh/nonuniform.xml")
mesh = Mesh("mesh/nonuniform_interval.xml")

bnd = 'dirichlet'
# bnd = 'periodic'

constr = PeriodicBoundary(Ld) if bnd is 'periodic' else None
V = FunctionSpace(mesh, 'CG', 1, constrained_domain=constr)

distr = Distributor(V, Ld, bnd)

plot(distr.dv)
