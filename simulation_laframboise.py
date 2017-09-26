from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
from punc import *

# Simulation parameters
tot_time = 1000                     # Total simulation time
dt       = 0.5                       # Time step
# vd       = np.array([0.0, 0.0])  # Drift velocity

# Get the mesh
mesh   = df.Mesh('mesh/lafram_coarse.xml')
Ld     = get_mesh_size(mesh)

# Create boundary conditions and function space
periodic = [False, False, False]
bnd = NonPeriodicBoundary(Ld, periodic)
constr = PeriodicBoundary(Ld, periodic)

V = df.FunctionSpace(mesh, 'CG', 1, constrained_domain=constr)

bc = df.DirichletBC(V, df.Constant(1.0), bnd)

# Get the object
class Probe(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and np.all(x<2)

objects = [Object(V, Probe())]

# Get the solver
poisson = PoissonSolver(V, bc)

# The inverse of capacitance matrix
inv_cap_matrix = capacitance_matrix(V, poisson, bnd, objects)

# Probe radius in terms of Debye lengths
Rp = 5.

Vnorm = Rp**2
#Inorm = -np.sqrt(8*np.pi/1836.)/Rp
Inorm = np.sqrt(8*np.pi)/Rp

# Initialize particle positions and velocities, and populate the domain
pop = Population(mesh, periodic)
pop.init_new_specie('electron', normalization='particle scaling', v_thermal=1./Rp, num_per_cell=4)
pop.init_new_specie('proton',   normalization='particle scaling', v_thermal=1./(np.sqrt(1836.)*Rp), num_per_cell=4)

dv_inv = voronoi_volume_approx(V, Ld)

injection = []
num_total = 0
# n_plasma = [None]*2

for i in range(len(pop.species)):
    # n_plasma[i] = pop.species[i].num_total
    # weight = pop.species.weight
    num_total += pop.species[i].num_total
    injection.append(Injector(pop, i, dt))

# Time loop
N   = tot_time
KE  = np.zeros(N-1)
PE  = np.zeros(N-1)
KE0 = kinetic_energy(pop)

current_collected = -1.869*Inorm
current_measured = np.zeros(N)
potential = np.zeros(N)

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

    old_charge = objects[0].charge
    pop.relocate(objects, open_bnd=True)
    #pop.relocate(objects)
    objects[0].add_charge(-current_collected*dt)
    current_measured[n] = ((objects[0].charge-old_charge)/dt)/Inorm
    potential[n] = objects[0]._potential/Vnorm

    # Inject particles:
    for inj in injection:
        inj.inject()

KE[0] = KE0

plt.plot(potential,label='potential')
# plt.plot(current_measured,label='current collected')
plt.legend(loc="lower right")
plt.show()

df.File('phi_laframboise.pvd') << phi
df.File('rho_laframboise.pvd') << rho
df.File('E_laframboise.pvd') << E

# df.plot(rho)
# df.plot(phi)

# ux = df.Constant((1,0,0))
# Ex = df.project(df.inner(E, ux), V)
# df.plot(Ex)
# df.interactive()
