from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import time
from punc import *
from scipy.spatial import Delaunay, Voronoi

preview = False

#==============================================================================
# GENERATE MESH
#------------------------------------------------------------------------------

print "Initializing solver"

Ld = 2*DOLFIN_PI*np.array([1,1])	# Length of domain
Nc = 8*np.array([1,1])				# Number of 'rectangles' in mesh

mesh = RectangleMesh(Point(0,0),Point(Ld),*Nc)

punc = Punc(mesh,PeriodicBoundary(Ld))

plot(mesh)
interactive()

print("Finished")
