#!/usr/bin/env python

# Copyright (C) 2017, Sigvald Marholm and Diako Darian
#
# This file is part of PUNC.
#
# PUNC is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# PUNC is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PUNC. If not, see <http://www.gnu.org/licenses/>.

"""
PUNC program for doing plasma-object interaction simulations, e.g. reproducing
the results of Laframboise. Usage:

    python object_interaction.py input_file.cfg.py

To force restarts:

    python object_interaction.py -r input_file.cfg.py

The input file specifies the geometry and all the simulation parameters. For
convenience it is fully Python-scriptable and has its own sandbox, i.e.
separate workspace from the rest of the program (this sandbox is not
unbreakable, but sufficient for a program only used by privileged users).
Certain variables from the configuration file is read as input.
"""

import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
from tasktimer import TaskTimer
from punc import *
import os
import signal
import sys
import logging

exit_now = False
def signal_handler(signal, frame):
    global exit_now
    if exit_now:
        sys.exit(0)
    else:
        print("\nCompleting and storing timestep before exiting. "
              "Press Ctrl+C again to force quit.")
        exit_now = True

def run():
    signal.signal(signal.SIGINT, signal_handler)
    df.set_log_level(logging.WARNING)

    # Loading input script in params (acting as a simplified sandbox)
    params = {}
    fname  = sys.argv[-1]
    code   = compile(open(fname, 'rb').read(), fname, 'exec')
    exec(code, params)

    # Loading input parameters
    object_method = params.pop('object_method', 'stiffness')
    dist_method   = params.pop('dist_method', 'voronoi')
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
    RLC           = params.pop('RLC', False)
    R             = params.pop('R', 0)
    L             = params.pop('L', 1e-6)
    C             = params.pop('C', 1e6)
    show_curr     = params.pop('show_curr', False)

    Ic = np.array([0., 0., 0.])
    Vc = np.array([0., 0.])
    tau1 = 0.5*dt*R/L
    tau2 = 0.5*dt**2/(L*C)

    assert object_method in ['capacitance', 'stiffness']
    assert dist_method in ['DG0', 'voronoi', 'weighted', 'patch', 'element']
    assert pe_method in ['mesh', 'particle']
    assert efield_method in ['project', 'evaluate', 'am', 'ci']
    # NB: only 'project' is tested as I do not have PETSC4py installed yet.

    V = df.FunctionSpace(mesh, 'CG', 1)
    Q = df.FunctionSpace(mesh, 'DG', 0)

    bc = df.DirichletBC(V, df.Constant(0.0), bnd, ext_bnd_id)

    if object_method=='capacitance':
        objects = [Object(V, bnd, int(i)) for i in int_bnd_ids]
        poisson = PoissonSolver(V, bc, eps0=eps0)
        inv_cap_matrix = capacitance_matrix(V, poisson, objects, bnd, ext_bnd_id)
        inv_cap_matrix /= cap_factor
    else:
        objects = [ObjectBC(V, bnd, i) for i in int_bnd_ids]
        circuit = Circuit(V, bnd, objects, vsources, isources, dt, eps0=eps0)
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
    if dist_method == 'voronoi':
        dv_inv = voronoi_volume_approx(V)
    elif dist_method == 'patch':
        dv_inv = patch_volume(V)
    elif dist_method == 'weighted':
        dv_inv = weighted_element_volume(V)

    continue_simulation = False
    if os.path.isfile('population.dat') and os.path.isfile('history.dat'):
        if '-r' in sys.argv:
            print("Overwriting existing simulation results.")
        else:
            print("Continuing previous simulation.")
            continue_simulation = True

    if continue_simulation:
        nstart, t = load_state('state.dat', objects)
        pop.load_file('population.dat')
        hist_file = open('history.dat', 'a')

    else:
        nstart, t = 0, 0.
        load_particles(pop, species)
        hist_file = open('history.dat', 'w')

    timer = TaskTimer()
    for n in timer.range(nstart, N):

        # We are now at timestep n
        # Velocities and currents are at timestep n-0.5 (or 0 if n==0)

        timer.task("Distribute charge ({})".format(dist_method))
        if dist_method=='voronoi' or dist_method=='patch' or dist_method=='weighted':
            rho = distribute(V, pop, dv_inv)
        elif dist_method == 'element':
            rho = distribute_elementwise(V, pop)
        elif dist_method == 'DG0':
            rho = distribute_DG0(Q, pop)

        timer.task("Solving potential ({})".format(object_method))
        if object_method == 'capacitance':

            # TBD: It would be nice if capacitance matrix method could be
            # re-implemented to support vsources and isources and thus be
            # completely interchangeable with stiffness matrix method. To be
            # planned a bit prior to execution.
            objects[0].charge -= collected_current*dt

            reset_objects(objects)
            phi = poisson.solve(rho, objects)
            E = esolve(phi)
            compute_object_potentials(objects, E, inv_cap_matrix, mesh, bnd)
            phi = poisson.solve(rho, objects)
        else:
            if RLC:
                Ic[1:0] = Ic[2:1]
                Vc[0] = Vc[1]
                Vc[1] = objects[1].potential - objects[0].potential
                Ic[2] = 2*Ic[1] - (1-tau1+tau2)*Ic[0] + (Vc[1]-Vc[0])*dt/L
                Ic[2] /= (1+tau1+tau2)
                isources[0][2] = Ic[2]
            if show_curr:
                print('RLC: {:g} {:g} {:g}'.format(R, L, C))
                print('Voltage: {:g}'.format(Vc[1]-Vc[0]))
                print('Current: {:g}'.format(isources[0][2]))
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
        KE = accel(pop, E, (1-0.5*(n==0))*dt)
        if n==0: KE = kinetic_energy(pop)

        timer.task("Write history")
        # Everything at n, except currents wich are at n-0.5.
        hist_write(hist_file, n, t, num_e, num_i, KE, PE, objects, Vnorm, Inorm)
        hist_file.flush()

        timer.task("Move particles")
        move(pop, dt) # Advancing position to n+1
        t += dt

        timer.task("Updating particles")
        pop.update(objects, dt)

        timer.task("Inject particles")
        inject_particles(pop, species, ext_bnd, dt)

        if exit_now or n==N-1:
            print("\n")
            save_state('state.dat', objects, n+1, t)
            pop.save_file('population.dat')
            break

    print(timer)
    hist_file.close()

    df.File('phi.pvd') << phi
    df.File('rho.pvd') << rho
    df.File('E.pvd')   << E
