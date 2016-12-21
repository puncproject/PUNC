from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import time
from punc import *
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
import pyvoro

preview = False

#==============================================================================
# GENERATE MESH
#------------------------------------------------------------------------------

print "Initializing solver"

Ld = 4.0*np.array([1,1])	# Length of domain
Nc = 4*np.array([1,1])		# Number of 'rectangles' in mesh

mesh = RectangleMesh(Point(0,0),Point(Ld),*Nc)
punc = Punc(mesh,PeriodicBoundary(Ld))

V = FunctionSpace(mesh,'CG',1)

x = project(Expression("x[0]",degree=1),V)
y = project(Expression("x[1]",degree=1),V)

xa = x.vector().array()
ya = y.vector().array()

points = np.array([xa,ya]).T

# Must remove periodic points here

vor = Voronoi(points,qhull_options="Fi")
#voronoi_plot_2d(vor)
#plt.show()

#allpoints = np.concatenate([points,vor.vertices])
allpoints=vor.vertices
allpoints[:,0] = np.minimum(allpoints[:,0],Ld[0]*np.ones(allpoints.shape[0]))
allpoints[:,0] = np.maximum(allpoints[:,0],0*np.ones(allpoints.shape[0]))
allpoints[:,1] = np.minimum(allpoints[:,1],Ld[1]*np.ones(allpoints.shape[0]))
allpoints[:,1] = np.maximum(allpoints[:,1],0*np.ones(allpoints.shape[0]))

dela = Delaunay(allpoints)
#plt.triplot(allpoints[:,0], allpoints[:,1], dela.simplices.copy())
#plt.plot(allpoints[:,0], allpoints[:,1], 'o')
#plt.show()

lower = 0-DOLFIN_EPS
upper = 3+DOLFIN_EPS

#points[:,0] = np.mod(points[:,0],upper)
#points[:,1] = np.mod(points[:,1],upper)

ran = [lower,upper]
vor2 = pyvoro.compute_2d_voronoi(points,[ran,ran],(upper-lower))


# Must somehow make cells periodic

#plot(mesh)
#interactive()

print("Finished")
