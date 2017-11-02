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
# vd       = np.array([0.0, 0.0])  # Drift velocity

# Get the mesh
sphere = SphereDomain()      # Create the sphereDomain object
mesh   = sphere.get_mesh()   # Get the mesh
Ld     = get_mesh_size(mesh) # Get the size of the simulation domain

# Create boundary conditions and function space
periodic = [False, False, False]
bnd = NonPeriodicBoundary(Ld, periodic)
constr = PeriodicBoundary(Ld, periodic)

V = df.FunctionSpace(mesh, 'CG', 1, constrained_domain=constr)

bc = df.DirichletBC(V, df.Constant(1.0), bnd)

# Get the circular object
objects = sphere.get_objects(V)

# Get the solver
poisson = PoissonSolver(V, bc)

# The inverse of capacitance matrix
inv_cap_matrix = capacitance_matrix(V, poisson, bnd, objects)

# Initialize particle positions and velocities, and populate the domain
pop = Population(mesh, periodic)
pop.init_new_specie('electron', temperature=1, num_per_cell=16)
pop.init_new_specie('proton',   temperature=1, num_per_cell=16)

dv_inv = voronoi_volume_approx(V)

# Time loop
N   = tot_time
KE  = np.zeros(N-1)
PE  = np.zeros(N-1)
KE0 = kinetic_energy(pop)

for n in range(1,N):
    print("Computing timestep %d/%d"%(n,N-1))

    rho = distribute(V, pop)
    compute_object_potentials(rho, objects, inv_cap_matrix)
    rho.vector()[:] *= dv_inv

    phi     = poisson.solve(rho, objects)
    E       = electric_field(phi)
    PE[n-1] = potential_energy(pop, phi)
    KE[n-1] = accel(pop, E, (1-0.5*(n==1))*dt)

    move_periodic(pop, Ld, dt)
    pop.relocate(objects)

KE[0] = KE0

df.File('phi_laframboise.pvd') << phi
df.File('rho_laframboise.pvd') << rho
df.File('E_laframboise.pvd') << E

# df.plot(rho)
# df.plot(phi)

# ux = df.Constant((1,0,0))
# Ex = df.project(df.inner(E, ux), V)
# df.plot(Ex)
# df.interactive()
