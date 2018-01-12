from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
from punc import *
import os
import sys

# NB: This is generally a discouraged technique, but allows great flexibility
# in the config file and is therefore used at least temporarily. That it is
# unsafe is not considered a problem since it's currently supposed to be used
# only by expert users.
exec('from %s import *'%(sys.argv[1]))

assert object_method in ['capacitance', 'variational']

# Get the mesh
mesh, bnd = load_mesh(fname)
ext_bnd_id, int_bnd_ids = get_mesh_ids(bnd)

V = df.FunctionSpace(mesh, 'CG', 1)

ext_bnd = ExteriorBoundaries(bnd, ext_bnd_id)
bc = df.DirichletBC(V, df.Constant(0.0), bnd, ext_bnd_id)

if object_method=='capacitance':
    objects = [Object(V, bnd, i) for i in int_bnd_ids]
else:
    objects = [Object(V, bnd, i) for i in int_bnd_ids]
    objects[0].set_potential(imposed_potential)
    objects[0].charge = 14.761509936666327

# Get the solver
poisson = PoissonSolver(V, bc, eps_0=eps_0)
esolver = ESolver(V)

if object_method=='capacitance':
    # The inverse of capacitance matrix
    inv_cap_matrix = capacitance_matrix(V, poisson, objects, bnd, ext_bnd_id)
    inv_cap_matrix /= cap_factor

species = SpeciesList(mesh, ext_bnd, X, T)
species.append('electron', n, vthe*w_pe, num=npc)
species.append('proton',   n, vthi*w_pe, num=npc)

pop = Population(mesh, bnd)
load_particles(pop, species)
nstart = 0
hist_file = open('history.dat', 'w')

# # Initialize particle positions and velocities, and populate the domain
# pop = Population(mesh, bnd, normalization=normtype)
# pop.species.X = X
# pop.species.T = T
# pop.species.D = D

# if os.path.isfile('stop'):
#     os.remove('stop')
#     pop.init_new_specie(electron, ext_bnd, v_thermal=vthe, num_per_cell=npc, empty=True)
#     pop.init_new_specie(ion     , ext_bnd, v_thermal=vthi, num_per_cell=npc, empty=True)
#     pop.load_file('population.dat')
#     f = open('state.dat','r')
#     nstart = int(f.readline()) + 1
#     objects[0].charge = float(f.readline())
#     f.close()
#     hist_file = open('history.dat', 'a')
# else:
#     nstart = 0
#     pop.init_new_specie(electron, ext_bnd, v_thermal=vthe, num_per_cell=npc)
#     pop.init_new_specie(ion     , ext_bnd, v_thermal=vthi, num_per_cell=npc)
#     hist_file = open('history.dat', 'w')

# boltzmann = 1.38064852e-23 # J/K
# pfreq =
# denorm = pop.species.get_denorm(pfreq, debye, debye)
# Vnorm = denorm['V']
# Inorm = denorm['I']

dv_inv = voronoi_volume_approx(V)

KE  = np.zeros(N-1)
PE  = np.zeros(N-1)

num_e = pop.num_of_negatives()
num_i = pop.num_of_positives()
potential = 0
current_measured = 0

timer = TaskTimer(N, 'compact')

for n in range(nstart, N):

    # We are now at timestep n
    # Velocities and currents are at timestep n-0.5 (or 0 if n==0)

    timer.task("Distribute charge")
    rho = distribute(V, pop, dv_inv)

    if object_method == 'capacitance':
        reset_objects(objects)

        timer.task("Solving potential 1")
        phi = poisson.solve(rho, objects)

        timer.task("Solving E-field 1")
        E = esolver.solve(phi)

        timer.task("Calculate object potential")
        compute_object_potentials(objects, E, inv_cap_matrix, mesh, bnd)
    else:
        pass

    potential = objects[0]._potential/Vnorm # at n

    timer.task("Solving potential 2")
    phi = poisson.solve(rho, objects)

    timer.task("Solving E-field 2")
    E = esolver.solve(phi)

    timer.task("Computing potential energy")
    # PE = particle_potential_energy(pop, phi)
    PE = mesh_potential_energy(rho, phi)

    timer.task("Move particles")
    old_charge = objects[0].charge # Charge at n
    KE = accel(pop, E, (1-0.5*(n==1))*dt) # Advancing velocities to n+0.5
    if n==0: KE = kinetic_energy(pop)
    move(pop, dt) # Advancing position to n+1

    timer.task("Updating particles")
    pop.update(objects)

    timer.task("Write history")
    hist_file.write("%d\t%d\t%d\t%f\t%f\t%f\t%f\n"%(n, num_e, num_i, KE, PE, potential, current_measured))
    hist_file.flush()

    timer.task("Impose current")
    current_measured = ((objects[0].charge-old_charge)/dt)/Inorm # at n+0.5
    # if object_method=='capacitance':
    objects[0].charge -= current_collected*dt

    timer.task("Inject particles")
    inject_particles(pop, species, ext_bnd, dt)

    timer.task("Count particles")
    num_i = pop.num_of_positives()
    num_e = pop.num_of_negatives()

    if os.path.isfile('stop') or n==N-1:
        pop.save_file('population.dat')
        f = open('state.dat','w')
        f.write("%d\n%f"%(n,objects[0].charge))
        f.close()
        break

    timer.end()

timer.summary()
hist_file.close()

df.File('phi.pvd') << phi
df.File('rho.pvd') << rho
df.File('E.pvd') << E
