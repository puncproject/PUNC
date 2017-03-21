from __future__ import print_function, division
from punc import *
import time
import numpy as np
import matplotlib.pyplot as plt
from dolfin import *
#==============================================================================
# SETTING UP DOMAIN
#------------------------------------------------------------------------------

nDims = 3
Ld = 2*np.pi*np.ones(nDims)
Nc = 16*np.ones(nDims,dtype=int)

if nDims==1: mesh = IntervalMesh(Nc[0],0.0,Ld[0])
if nDims==2: mesh = RectangleMesh(Point(0,0),Point(Ld),*Nc)
if nDims==3: mesh = BoxMesh(Point(0,0,0),Point(Ld),*Nc)

V = FunctionSpace(mesh, 'CG', 1, constrained_domain=PeriodicBoundary(Ld))
pop = Population(mesh)
rho = Function(V)

N = 100000
bins = 100

#==============================================================================
# CREATING PARTICLES
#------------------------------------------------------------------------------

# t = time.time()
A = 1.
pos = random_points(lambda x: 1+A*np.sin(2*x[0]),Ld,N,1+A)

plt.hist(pos[:,0],bins,facecolor='green',normed=1)
x = np.linspace(0,Ld[0],bins)
plt.plot(x,(1+A*np.sin(2*x))/(2*np.pi),'r-')
plt.show()

vd = np.zeros(nDims)
vth = 1
vel = np.random.normal(vd,vth*np.ones(nDims),pos.shape)
speed = [np.sqrt(np.sum(v**2)) for v in vel]
plt.hist(speed,bins,normed=1)
v = np.linspace(0,6,bins)

y = (1/(2*np.pi*vth**2))**(nDims/2.0)*np.exp(-0.5*(v/vth)**2)
if nDims==2: y *= 2*np.pi*v
if nDims==3: y *= 4*np.pi*v**2

plt.plot(v,y,'r-')
plt.show()

posIon = random_points(lambda x: 0.5, Ld, N)
velIon = np.array([0,0,0]) # Cold ions

mul = np.prod(Ld)/N

pop.add_particles(pos,vel,-mul,mul)
pop.add_particles(posIon,velIon,mul,500*mul)

#==============================================================================
# COMPUTING FIELDS
#------------------------------------------------------------------------------

#pop.distr(rho)
#plot(rho)

dv_inv = voronoi_volume(V,Ld)
rho = distribute(V,pop)
rho.vector()[:] *= dv_inv
plot(rho)
interactive()
