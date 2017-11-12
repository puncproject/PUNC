from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np
from punc import *

from matplotlib import pyplot as plt

# Simulation parameters
tot_time = 100                   # Total simulation time
dim      = 2
dt       = 0.1              # Time step
v_thermal = .2

debug = True
plot = True

if dim == 2:
    v_drift  = np.array([0.0, 0.0])  # Drift velocity
    Ld = [6.0, 6.0]
    N = [16, 16]
elif dim == 3:
    v_drift  = np.array([0.0, 0.0, 0.0])  # Drift velocity
    Ld = [np.pi, np.pi, np.pi]
    N = [5,5,5]

# Get the mesh
mesh, facet_func = simple_mesh(Ld, N) # Get the mesh
ext_bnd_id, int_bnd_ids = get_mesh_ids(facet_func)

Ld = get_mesh_size(mesh)  # Get the size of the simulation domain

exterior_bnd = ExteriorBoundaries(facet_func, ext_bnd_id)

# Initialize particle positions and velocities, and populate the domain
pop = Population(mesh, facet_func, normalization='none')
pop.init_new_specie('electron', exterior_bnd, v_drift=v_drift,
                    v_thermal=v_thermal, num_per_cell=32)
pop.init_new_specie('proton', exterior_bnd, v_drift=v_drift,
                    v_thermal=v_thermal, num_per_cell=32)

# Time loop
N   = tot_time
KE  = np.zeros(N)
KE0 = kinetic_energy(pop)
num_particles = np.zeros(N)
num_particles_outside = np.zeros(N)
num_injected_particles = np.zeros(N)
num_particles[0] = pop.num_of_particles()
print("num_particles: ", num_particles[0])

num_e = np.zeros(N)
num_i = np.zeros(N)
num_e[0] = num_particles[0] / 2
num_i[0] = num_particles[0] / 2

for n in range(1,N):
    if debug:
        print("Computing timestep %d/%d"%(n,N-1))
   
    # Total number of particles before injection:
    tot_num0 = pop.num_of_particles()
   
    # Move the particles:
    move(pop,dt)
   
    # Update particle positions:
    pop.update()

    tot_num1 = pop.num_of_particles()
    # Total number of particles leaving the domain:
    num_particles_outside[n] = tot_num0 -tot_num1

    # Inject particles:
    inject(pop, exterior_bnd, dt)

    tot_num2 = pop.num_of_particles()
    # Total number of injected particles:
    num_injected_particles[n] = tot_num2 - tot_num1

    # Number of ions and electrons in the domian.
    num_i[n] = pop.num_of_positives()
    num_e[n] = pop.num_of_negatives()
    # The total kinetic energy:
    KE[n] = kinetic_energy(pop)

    # Total number of particles in the domain
    num_particles[n] = pop.num_of_particles()
    if debug:
        print("Total number of particles in the domain: ", tot_num2)

KE[0] = KE0

if plot:
    to_file = open('injection.txt', 'w')
    for i,j,k,l in zip(num_particles, num_injected_particles, num_particles_outside, KE):
        to_file.write("%f %f %f %f\n" %(i, j, k, l))
    to_file.close()

    plt.figure()
    plt.plot(num_particles,label="Total number of particles")
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
    
    plt.figure()
    plt.plot(num_i, label="Number of ions")
    plt.plot(num_e, label="Number of electrons")
    plt.legend(loc='lower right')
    plt.grid()
    plt.xlabel("Timestep")
    plt.ylabel("Number of particles")
    plt.savefig('e_i_numbers.png', format='png', dpi=1000)

    plt.show()
