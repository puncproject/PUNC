from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import time
from punc import *
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
import pyvoro

#==============================================================================
# GENERATE MESH
#------------------------------------------------------------------------------

nDims = 2

Ld = 3.0*np.ones(nDims)			# Length of domain
Nc = 3*np.ones(nDims,dtype=int)	# Number of 'rectangles' in mesh

if nDims==2:
	mesh = RectangleMesh(Point(0,0),Point(Ld),*Nc)
if nDims==3:
	mesh = BoxMesh(Point(0,0,0),Point(Ld),*Nc)

punc = Punc(mesh,PeriodicBoundary(Ld))

V = FunctionSpace(mesh,'CG',1)

limits = np.zeros([nDims,2]) - DOLFIN_EPS
limits[:,1] = Ld             + DOLFIN_EPS

nVertices = V.dim()
		
vertices = np.zeros([nVertices,nDims])
for i in range(nDims):
	expr = Expression("x[%d]"%i, degree=1)
	vertices[:,i] = project(expr, V).vector().array()

# Must remove periodic points here

if nDims==2:
	vor = pyvoro.compute_2d_voronoi(vertices,limits,2)
if nDims==3:
	vor = pyvoro.compute_voronoi(vertices,limits,2)



print("Finished")
