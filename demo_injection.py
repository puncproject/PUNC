from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np
from punc import *

from mesh import *

# Simulation parameters
tot_time = 20                    # Total simulation time
dt       = 0.251327              # Time step
vd       = np.array([0.0, 0.0])  # Drift velocity

# Get the mesh
Ld = [2*np.pi, 2*np.pi]
N = [32, 32]
mesh = simple_mesh(Ld, N) # Get the mesh
Ld = get_mesh_size(mesh)  # Get the size of the simulation domain

# Create boundary conditions and function space
periodic = [False, False, False]
bnd      = NonPeriodicBoundary(Ld, periodic)
V        = df.FunctionSpace(mesh, "CG", 1)

# Get the solver
bc      = df.DirichletBC(V, df.Constant(0), bnd)
poisson = PoissonSolver(V, bc)

# Initialize particle positions and velocities, and populate the domain
pop = Population(mesh)
pop.init_new_specie('electron', temperature=1)
pop.init_new_specie('proton'  , temperature=1)

dv_inv = voronoi_volume(V, Ld, periodic)

# To be replaced by conostructor for Injector.
pdf = [lambda x: 1, lambda x: 1]
init = Initialize(pop, pdf, Ld, [0,0], [1,1], 8, dt = dt)

# Time loop
N   = tot_time
KE  = np.zeros(N-1)
PE  = np.zeros(N-1)
KE0 = kinetic_energy(pop)

for n in range(1,N):
    print("Computing timestep %d/%d"%(n,N-1))

    rho = distribute(V, pop)
    rho.vector()[:] *= dv_inv

    phi     = poisson.solve(rho, bc)
    E       = electric_field(phi)
    PE[n-1] = potential_energy(pop, phi)
    KE[n-1] = accel(pop,E,(1-0.5*(n==1))*dt)

    move(pop, Ld, dt)

    pop.relocate(open_bnd = True)

    tot_p = pop.total_number_of_particles()
    print("Total number of particles in the domain: ", tot_p)

    init.inject()

    tot_p = pop.total_number_of_particles()
    print("Total number of particles in the domain: ", tot_p)

KE[0] = KE0

df.plot(rho)
df.plot(phi)
#
ux = df.Constant((1,0))
Ex = df.project(df.inner(E, ux), V)
df.plot(Ex)
df.interactive()

# df.File("rho.pvd") << rho
# df.File("phi.pvd") << phi
# df.File("E.pvd") << E
