from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np
from punc import *

from mesh import *
from parameters import *


# Simulation parameters
tot_time = 20                    # Total simulation time
dt       = 0.251327              # Time step
vd       = np.array([0.0, 0.0])  # Drift velocity

# Get the mesh
circles = CircuitDomain()      # Create the CircleDomain object
mesh    = circles.get_mesh()   # Get the mesh
Ld      = get_mesh_size(mesh)  # Get the size of the simulation domain

# Create boundary conditions and function space
periodic = [True, True, True]
PBC      = PeriodicBoundary(Ld)
V        = df.FunctionSpace(mesh, "CG", 1, constrained_domain=PBC)

# Get the solver
poisson = PoissonSolver(V, remove_null_space=True)

# Create objects
objects = circles.get_objects(V)

# The inverse of capacitance matrix
inv_cap_matrix = capacitance_matrix(V, poisson, objects)

# Create the circuits
circuits = circles.get_circuits(objects, inv_cap_matrix)

# Initialize particle positions and velocities, and populate the domain
pop    = Population(mesh)
dv_inv = voronoi_volume(V, Ld, periodic)

pdf  = [lambda x: 1, lambda x: 1]
init = Initialize(pop, pdf, Ld, vd, [alpha_e,alpha_i], 8, objects=objects)
init.initial_conditions()

# Time loop
N   = tot_time
KE  = np.zeros(N-1)
PE  = np.zeros(N-1)
KE0 = kineticEnergy(pop)

for n in range(1,N):
    print("Computing timestep %d/%d"%(n,N-1))

    rho = distribute(V, pop)
    compute_object_potentials(rho, objects, inv_cap_matrix)
    rho.vector()[:] *= dv_inv

    phi     = poisson.solve(rho, objects)
    E       = electric_field(phi)
    PE[n-1] = potentialEnergy(pop, phi)
    KE[n-1] = accel(pop, E, (1-0.5*(n==1))*dt)

    movePeriodic(pop, Ld, dt)
    pop.relocate(objects)

    redistribute_circuit_charge(circuits)

KE[0] = KE0

df.plot(rho)
df.plot(phi)

ux = df.Constant((1,0))
Ex = df.project(df.inner(E, ux), V)
df.plot(Ex)
df.interactive()
