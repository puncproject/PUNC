from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import numpy as np
from punc import *

import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

E0 = 0.1
a = .5
mesh = df.BoxMesh(df.Point(0, 0,0), df.Point(2, 2,2), 10, 10,5)
Ld = [2., 2., 2.]
xs = np.array([[1.0,0.05, 0.5]])
vs = np.array([[0., 0., 0.0]])
q = -1.
m = .1
Np = 1*mesh.num_cells()
mul = (np.prod(Ld)/np.prod(Np))

pop = Population(mesh)
pop.addParticles(xs,vs,q*mul,m*mul)

V = df.VectorFunctionSpace(mesh, 'CG', 1)
E = df.interpolate(df.Expression(("E0*(x[0]-1)/pow(pow((x[0]-1),2)+pow((x[1]-1),2),1.5)",
                                  "E0*(x[1]-1)/pow(pow((x[0]-1),2)+pow((x[1]-1),2),1.5)", "0"),
                                   E0=E0, degree=3),V)

B0 = df.interpolate(df.Expression(("0", "0",
                                  "a*pow(pow((x[0]-1),2)+pow((x[1]-1),2),0.5)"),
                                   a = a, degree=3),V)

# df.plot(B0)
# df.plot(E)
# df.interactive()
#-------------------------------------------------------------------------------
#             Time loop
#-------------------------------------------------------------------------------
N = 220#3800
dt = 0.1
KE = np.zeros(N-1)
KE0 = kineticEnergy(pop)
pos = np.zeros((N,3))
pos[0] = xs
for n in range(1,N):
    print("t: ", n)
    KE[n-1] = accel(pop,E,(1-0.5*(n==1))*dt, B0)
    q_object = movePeriodic(pop, Ld, dt)

    for cell in pop:
        for particle in cell:
            pos[n] = particle.x
KE[0] = KE0
print(pos)

fig = plt.figure()
plt.plot(pos[:,0],pos[:,1])
plt.axis([0, 2, 0, 2])
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
