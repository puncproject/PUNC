from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import numpy as np
from punc import *
import matplotlib.pyplot as plt

E0 = 0.007  # The strength of the electric field
B0 = 1.     # The strength of the magnetic field

Ld = [2., 2., 2.]               # The simulation domain
N = [10, 10, 5]

# Get the mesh
mesh, facet_func = simple_mesh(Ld, N)  # Get the mesh
ext_bnd_id, int_bnd_ids = get_mesh_ids(facet_func)
Ld = get_mesh_size(mesh)  # Get the size of the simulation domain

exterior_bnd = ExteriorBoundaries(facet_func, ext_bnd_id)

xs = np.array([[1.0, 0.5, 0.5]]) # Initial position
vs = np.array([[0.1, 0., 0.]])   # Initial velocity

q = -1. # Particle Charge
m = .05 # Particle mass
Np = 1*mesh.num_cells()
mul = (np.prod(Ld)/np.prod(Np))

pop = Population(mesh, facet_func, normalization='none')
pop.add_particles(xs,vs,q*mul,m*mul) # Add particle to population

V = df.VectorFunctionSpace(mesh, 'CG', 1) # The vector function space

# The electric field
E = df.interpolate(df.Expression(( "E0*(x[0]-1)/pow(pow((x[0]-1),2)+pow((x[1]-1),2),1.5)",
                                  "E0*(x[1]-1)/pow(pow((x[0]-1),2)+pow((x[1]-1),2),1.5)", "0"),
                                   E0=E0, degree=3),V)
# The magnetic field
B = df.interpolate(df.Expression(("0", "0",
                                  "B0*pow(pow((x[0]-1),2)+pow((x[1]-1),2),0.5)"),
                                   B0 = B0, degree=3),V)

# df.plot(B)
# df.plot(E)
# df.interactive()
#-------------------------------------------------------------------------------
#             Time loop
#-------------------------------------------------------------------------------
N = 300#4580           # Number of time steps
dt = .01           # Time step
KE = np.zeros(N-1)
KE0 = kinetic_energy(pop)
pos = np.zeros((N,3)) # Particle positions
pos[0] = xs
for n in range(1,N):
    print("t: ", n)
    KE[n-1] = boris(pop,E,B,(1-0.5*(n==1))*dt)
    move_periodic(pop, Ld, dt)
    pop.update()

    for cell in pop:
        for particle in cell:
            pos[n] = particle.x
KE[0] = KE0

fig = plt.figure()
plt.plot(pos[:,0],pos[:,1], label='Particle trajectory')
plt.xlim([0,2])
plt.ylim([0,2])
plt.grid()
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right')
plt.show()
