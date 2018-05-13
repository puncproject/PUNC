"""
PUNC program for doing plasma-object interaction simulations, e.g. reproducing
the results of Laframboise. Usage:

    python object_interaction.py input_file.cfg.py

The input file specifies the geometry and all the simulation parameters. For
convenience it is fully Python-scriptable and has its own sandbox, i.e.
separate workspace from the rest of the program (this sandbox is not
unbreakable, but sufficient for a program only used by privileged users).
Certain variables from the configuration file is read as input.
"""
# For now, just put ConstantBC and punc directories in PYTHONPATH

import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
from punc import *
from ConstantBC import *
from ConstantBC import Circuit as Circ
import os
import signal

exit_now = False
def signal_handler(signal, frame):
    global exit_now
    if exit_now:
        sys.exit(0)
    else:
        print("\nCompleting and storing timestep before exiting. "
              "Press Ctrl+C again to force quit.")
        exit_now = True

signal.signal(signal.SIGINT, signal_handler)

df.set_log_level(df.WARNING)

# Loading input script in params (acting as a simplified sandbox)
params = {}
fname  = sys.argv[1]
code   = compile(open(fname, 'rb').read(), fname, 'exec')
exec(code, params)

# Loading input parameters
object_method = params.pop('object_method', 'stiffness')
dist_method   = params.pop('dist_method', 'element')
pe_method     = params.pop('pe_method', 'mesh')
efield_method = params.pop('efield_method', 'project')
mesh          = params.pop('mesh')
bnd           = params.pop('bnd')
ext_bnd       = params.pop('ext_bnd')
ext_bnd_id    = params.pop('ext_bnd_id', 1)
int_bnd_ids   = params.pop('int_bnd_ids')
species       = params.pop('species')
eps0          = params.pop('eps0', 1)
cap_factor    = params.pop('cap_factor', 1)
dt            = params.pop('dt')
N             = params.pop('N')
vsources      = params.pop('vsources', None)
isources      = params.pop('isources', None)
Vnorm         = params.pop('Vnorm', 1)
Inorm         = params.pop('Inorm', 1)

assert object_method in ['capacitance', 'stiffness']
assert dist_method in ['voronoi', 'element']
assert pe_method in ['mesh', 'particle']
assert efield_method in ['project', 'evaluate', 'am', 'ci']
# NB: only 'project' is tested as I do not have PETSC4py installed yet.

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

if efield_method == 'project':
    esolver = ESolver(V)
    esolve = esolver.solve
elif efield_method == 'evaluate':
    esolve = lambda phi: efield_DG0(mesh, phi)
elif efield_method == 'am':
    esolver = EfieldMean(mesh, arithmetic_mean=True)
    esolve = esolver.mean
elif efield_method == 'ci':
    esolver = EfieldMean(mesh, arithmetic_mean=False)
    esolve = esolver.mean

pop = Population(mesh, bnd)
dv_inv = voronoi_volume_approx(V)

if os.path.isfile('stop'):
    os.remove('stop')
    nstart = hist_last_step('history.dat') + 1
    hist_load('history.dat', objects)
    hist_file = open('history.dat', 'a')
    pop.load_file('population.dat')

else:
    nstart = 0
    load_particles(pop, species)
    hist_file = open('history.dat', 'w')

KE  = np.zeros(N-1)
PE  = np.zeros(N-1)

num_e = pop.num_of_negatives()
num_i = pop.num_of_positives()

timer = TaskTimer(N, 'compact')

t = 0.
for n in range(nstart, N):

    # We are now at timestep n
    # Velocities and currents are at timestep n-0.5 (or 0 if n==0)

    timer.task("Distribute charge ({})".format(dist_method))
    if dist_method == 'voronoi':
        rho = distribute(V, pop, dv_inv)
    else:
        rho = distribute_elementwise(V, pop)

    timer.task("Solving potential ({})".format(object_method))
    if object_method == 'capacitance':

        # TBD: It would be nice if capacitance matrix method could be
        # re-implemented to support vsources and isources and thus be
        # completely interchangeable with stiffness matrix method. To be
        # planned a bit prior to execution.
        objects[0].charge -= current_collected*dt

        reset_objects(objects)
        phi = poisson.solve(rho, objects)
        E = esolve(phi)
        compute_object_potentials(objects, E, inv_cap_matrix, mesh, bnd)
        phi = poisson.solve(rho, objects)
    else:
        phi = poisson.solve(rho)
        for o in objects:
            o.update(phi)

    timer.task("Solving E-field ({})".format(efield_method))
    E = esolve(phi)

    timer.task("Computing potential energy ({})".format(pe_method))
    if pe_method == 'particle':
        PE = particle_potential_energy(pop, phi)
    else:
        PE = mesh_potential_energy(rho, phi)

    timer.task("Count particles")
    num_i = pop.num_of_positives()
    num_e = pop.num_of_negatives()

    timer.task("Accelerate particles")
    # Advancing velocities to n+0.5
    KE = accel(pop, E, (1-0.5*(n==1))*dt)
    if n==0: KE = kinetic_energy(pop)

    timer.task("Write history")
    # Everything at n, except currents wich are at n-0.5.
    hist_write(hist_file, n, num_e, num_i, KE, PE, objects, Vnorm, Inorm)
    hist_file.flush()

    timer.task("Move particles")
    move(pop, dt) # Advancing position to n+1
    t += dt

    timer.task("Updating particles")
    pop.update(objects, dt)

    timer.task("Inject particles")
    inject_particles(pop, species, ext_bnd, dt)


    if os.path.isfile('stop') or exit_now or n==N-1:
        pop.save_file('population.dat')
        # f = open('state.dat','w')
        # f.write("%d\n%f"%(n,objects[0].charge))
        # f.close()
        break

    timer.end()

timer.summary()
hist_file.close()

df.File('phi.pvd') << phi
df.File('rho.pvd') << rho
df.File('E.pvd')   << E
