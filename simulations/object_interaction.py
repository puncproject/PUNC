# For now, just put ConstantBC and punc directories in PYTHONPATH

import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
from punc import *
from ConstantBC import *
from ConstantBC import Circuit as Circ
import os

df.set_log_level(df.WARNING)

# NB: This is generally a discouraged technique, but allows great flexibility
# in the config file and is therefore used at least temporarily. That it is
# unsafe is not considered a problem since it's currently supposed to be used
# only by expert users.
exec('from %s import *'%(sys.argv[1]))

assert object_method in ['capacitance', 'stiffness']

V = df.FunctionSpace(mesh, 'CG', 1)

bc = df.DirichletBC(V, df.Constant(0.0), bnd, ext_bnd_id)

if object_method=='capacitance':
    objects = [Object(V, bnd, int(i)) for i in int_bnd_ids]
    poisson = PoissonSolver(V, bc, eps0=eps0)

    inv_cap_matrix = capacitance_matrix(V, poisson, objects, bnd, ext_bnd_id)
    inv_cap_matrix /= cap_factor

else:
    objects = [ObjectBC(V, bnd, i) for i in int_bnd_ids]
    circuit = Circ(V, bnd, objects, vsources, isources, dt, eps0=eps0)
    poisson = PoissonSolver(V, bc, objects, circuit, eps0=eps0)
    # objects = [df.DirichletBC(V, df.Constant(1), bnd, i) for i in int_bnd_ids]
    # poisson = PoissonSolver(V, bc, objects, eps0=eps0)

esolver = ESolver(V)
pop = Population(mesh, bnd)
dv_inv = voronoi_volume_approx(V)

if os.path.isfile('stop'):
    os.remove('stop')
    pop.load_file('population.dat')
    f = open('state.dat','r')
    nstart = int(f.readline()) + 1
    objects[0].charge = float(f.readline())
    f.close()
    hist_file = open('history.dat', 'a')

else:
    load_particles(pop, species)
    nstart = 0
    hist_file = open('history.dat', 'w')

KE  = np.zeros(N-1)
PE  = np.zeros(N-1)

num_e = pop.num_of_negatives()
num_i = pop.num_of_positives()
potential = 0
current_measured = 0

timer = TaskTimer(N, 'simple')

for n in range(nstart, N):

    # We are now at timestep n
    # Velocities and currents are at timestep n-0.5 (or 0 if n==0)

    timer.task("Distribute charge")
    rho = distribute(V, pop, dv_inv)

    timer.task("Solving potential ({})".format(object_method))
    if object_method == 'capacitance':
        reset_objects(objects)
        phi = poisson.solve(rho, objects)
        E = esolver.solve(phi)
        compute_object_potentials(objects, E, inv_cap_matrix, mesh, bnd)
        potential = objects[0]._potential*Vnorm # at n
        phi = poisson.solve(rho, objects)
    else:
        phi = poisson.solve(rho)
        # for o in objects:
        #     o.correct_charge(phi)
        potential = objects[0].get_potential(phi)*Vnorm

    timer.task("Solving E-field")
    E = esolver.solve(phi)

    timer.task("Computing potential energy")
    PE = particle_potential_energy(pop, phi)
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
    current_measured = ((objects[0].charge-old_charge)/dt)*Inorm # at n+0.5
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
df.File('E.pvd')   << E
