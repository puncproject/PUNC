from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np
from punc import *
import time
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt

df.set_log_active(False)

# Simulation parameters
tot_time = 100                   # Total simulation time
dim      = 2
dt       = 0.1              # Time step
k = 2.0
v_thermal = 1.0

debug = True
plot = True

if dim == 2:
    v_drift  = np.array([0.0, 0.0])  # Drift velocity
    Ld = [6.0, 6.0]
    N = [32, 32]
elif dim == 3:
    v_drift  = np.array([0.0, 0.0, 0.0])  # Drift velocity
    Ld = [np.pi, np.pi, np.pi]
    N = [5,5,5]

# Get the mesh
# mesh, facet_func = load_mesh("../mesh/2D/ellipse")
mesh, facet_func = simple_mesh(Ld, N) # Get the mesh
ext_bnd_id, int_bnd_ids = get_mesh_ids(facet_func)

Ld = get_mesh_size(mesh)  # Get the size of the simulation domain

ext_bnd = ExteriorBoundaries(facet_func, ext_bnd_id)

# Initialize particle positions and velocities, and populate the domain
npc = 8
me = 1.0#constants.value('electron mass')
e = 1.0#constants.value('elementary charge')
ne = 1e6
X = np.mean(Ld)

vdf_type = 'kappa'

species = SpeciesList(mesh, ext_bnd, X)
species.append_raw(-e, me, ne, v_thermal, v_drift, npc=npc, k=k,vdf_type=vdf_type)

pop = Population(mesh, facet_func)

load_particles(pop, species)

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
    inject_particles(pop, species, ext_bnd, dt)

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
    vs = []
    for cell in pop:
        for particle in cell:
            vs.append(particle.v)
    vs = np.array(vs)        
    vth = species[0].vth 
    vd = species[0].vd
    def pdf_maxwellian(i, t):
        return 1.0 / (np.sqrt(2 * np.pi) * vth) *\
            np.exp(-0.5 * ((t - vd[i])**2) / (vth**2))

    def pdf_kappa(i, t):
        return 1.0 / ((np.pi * (2 * k - 3.) * vth**2)**((dim - 1) / 2.0)) *\
            ((gamma(k + 0.5 * ((dim - 1) - 1.0))) / (gamma(k - 0.5))) *\
            (1. + (t - vd[i])**2 / ((2 * k - 3.) * vth**2)
                )**(-(k + 0.5 * ((dim - 1) - 1.)))
                
    xs = np.linspace(vd[0] - 5 * vth, vd[0] + 5 * vth, 1000)

    plt.figure(figsize=(8, 7))
    plt.hist2d(vs[:,0],vs[:,1],bins=100,norm=LogNorm())
    plt.figure(figsize=(10, 7))
    plt.hist(vs[:,0], bins=300, color = 'blue', normed=1)
    if vdf_type=='maxwellian':
        plt.plot(xs, pdf_maxwellian(0,xs), color='red')
    elif vdf_type=='kappa':
        plt.plot(xs, pdf_kappa(0, xs), color='red')
    plt.figure(figsize=(10, 7))
    plt.hist(vs[:,1], bins=300, color = 'blue', normed=1)
    if vdf_type == 'maxwellian':
        plt.plot(xs, pdf_maxwellian(1, xs), color='red')
    elif vdf_type == 'kappa':
        plt.plot(xs, pdf_kappa(1, xs), color='red')
    plt.show()
    
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
