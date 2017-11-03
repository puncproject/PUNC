from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np
from punc import *

from mesh import *
from matplotlib import pyplot as plt

# Simulation parameters
tot_time = 1000                   # Total simulation time
dim      = 2
dt       = 0.001              # Time step
v_thermal = 1.0

debug = True
plot = True

if dim == 2:
    v_drift  = np.array([0.0, 0.0])  # Drift velocity
    Ld = [2.0, 2.0]
    N = [30, 30]
    periodic = [False, False]
elif dim == 3:
    v_drift  = np.array([0.0, 0.0, 0.0])  # Drift velocity
    Ld = [np.pi, np.pi, np.pi]
    N = [5,5,5]
    periodic = [False, False, False]

# Get the mesh
mesh = simple_mesh(Ld, N) # Get the mesh
Ld = get_mesh_size(mesh)  # Get the size of the simulation domain

# Create boundary conditions and function space
# bnd      = NonPeriodicBoundary(Ld, periodic)
# V        = df.FunctionSpace(mesh, "CG", 1)


# Initialize particle positions and velocities, and populate the domain
pop = Population(mesh, periodic)
pop.init_new_specie('electron', v_drift=v_drift, v_thermal=v_thermal, num_per_cell=32)
pop.init_new_specie('proton', v_drift=v_drift, v_thermal=v_thermal, num_per_cell=32)

mv = Move(pop, dt)

volume = df.assemble(1*df.dx(mesh))

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
KE  = np.zeros(N)
KE0 = kinetic_energy(pop)

num_particles = np.zeros(N)
num_particles_outside = np.zeros(N)
num_injected_particles = np.zeros(N)

num_particles[0] = pop.total_number_of_particles()[0]/volume

for n in range(1,N):
    if debug:
        print("Computing timestep %d/%d"%(n,N-1))
    # Total number of particles before injection:
    tot_num0 = pop.total_number_of_particles()[0]
    # Move the particles:
    # move(pop, Ld, dt)
    mv.move()
    # Relocate particles:
    pop.relocate(open_bnd = True)

    tot_num1 = pop.total_number_of_particles()[0]
    # Total number of particles leaving the domain:
    num_particles_outside[n] = tot_num0 -tot_num1

    # Inject particles:
    for inj in injection:
        inj.inject()

    tot_num2 = pop.total_number_of_particles()[0]
    # Total number of injected particles:
    num_injected_particles[n] = tot_num2 - tot_num1
    # Total number of particles after injection:
    num_particles[n] = tot_num2/volume
    # The total kinetic energy:
    KE[n] = kinetic_energy(pop)
    if debug:
        print("Total number of particles in the domain: ", tot_num2)

KE[0] = KE0

if plot:
    to_file = open('injection.txt', 'w')
    for i,j,k,l in zip(num_particles, num_injected_particles, num_particles_outside, KE):
        to_file.write("%f %f %f %f\n" %(i, j, k, l))
    to_file.close()

    plt.figure()
    plt.plot(num_particles,label="Total number denisty")
    plt.legend(loc='lower right')
    plt.grid()
    plt.xlabel("Timestep")
    plt.ylabel("Total number denisty")
    plt.savefig('total_num.png')

    plt.figure()
    plt.plot(num_injected_particles[1:], label="Number of injected particles")
    plt.plot(num_particles_outside[1:], label="Number of particles leaving the domain")
    plt.legend(loc='lower right')
    plt.grid()
    plt.xlabel("Timestep")
    plt.ylabel("Number of particles")
    plt.savefig('injected.png')

    plt.figure()
    plt.plot(KE,label="Kinetic Energy")
    plt.legend(loc='lower right')
    plt.grid()
    plt.xlabel("Timestep")
    plt.ylabel("Normalized Energy")
    plt.savefig('kineticEnergy.png')
    plt.show()
