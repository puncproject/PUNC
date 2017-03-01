from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import time

import sys
#sys.path.append('../src')
#from punc import *

from punc import *

if sys.version_info.major == 2:
	range = xrange

#==============================================================================
# GENERATE MESH
#------------------------------------------------------------------------------

print "Initializing solver"

Ld = 6.28*np.array([1,1])			# Length of domain
Nc = 32*np.array([1,1])				# Number of 'rectangles' in mesh
mesh = RectangleMesh(Point(0,0),Point(Ld),*Nc)
#mesh = Mesh("mesh/nonuniform.xml")

#==============================================================================
# SOLVER
#------------------------------------------------------------------------------

solver = PoissonSolver(mesh,Ld,PeriodicBoundary(Ld))


