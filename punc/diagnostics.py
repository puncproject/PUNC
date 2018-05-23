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
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PUNC.  If not, see <http://www.gnu.org/licenses/>.
import dolfin as df
import numpy as np

def hist_load(fname, objects):

    with open(fname, 'r') as fh:
        for line in fh:
            pass # Get the last line
        values = line.split()

    for i,o in enumerate(objects):
        o.charge = float(values[3*i+6])

def hist_last_step(fname):

    with open(fname, 'r') as fh:
        for line in fh:
            pass # Get the last line
        values = line.split()

    return int(values[0])

def hist_write(fh, n, t=0, num_e=0, num_i=0, KE=0, PE=0, objects=[],
                  Vnorm=1, Inorm=1):
    fh.write("%d\t%f\t%d\t%d\t%f\t%f"%(n, t, num_e, num_i, KE, PE))
    for o in objects:
        fh.write("\t%f\t%f\t%f"%(o.charge,
                                 o.potential*Vnorm,
                                 o.collected_current*Inorm))
    fh.write("\n")

def kinetic_energy(pop):
    """
    Computes kinetic energy at current velocity time step.
    Useful for the first (zeroth) time step before velocity has between
    advanced half a timestep. To get velocity between two velocity time
    steps (e.g. at integer steps after the start-up) use accel() return value.
    """
    KE = 0
    for cell in pop:
        for particle in cell:
            m = particle.m
            v = particle.v
            KE += 0.5*m*np.dot(v,v)
    return KE

def mesh_potential_energy(rho, phi):
    """
    Computes potential energy at current time step from mesh quantities. Should
    be equivalent to particle_potential_energy() to within numerical accuracy
    but faster by orders of magnitude. It remains to determine whether this or
    particle_potential_energy() or neither is correct in the precense of objects.
    """

    return 0.5*df.assemble(rho*phi*df.dx)

def efield_potential_energy(E):

    return 0.5*df.assemble(df.dot(E,E)*df.dx)

def particle_potential_energy(pop ,phi):
    """
    Computes potential energy at current time step from particles. Should
    be equivalent to mehs_potential_energy() to within numerical accuracy
    but slower by orders of magnitude. It remains to determine whether this or
    mesh_potential_energy() or neither is correct in the precense of objects.
    """

    PE = 0

    V = phi.function_space()
    element = V.dolfin_element()
    s_dim = element.space_dimension()  # Number of nodes per element
    basis_matrix = np.zeros((s_dim,1))
    coefficients = np.zeros(s_dim)

    for cell in df.cells(pop.mesh):
        phi.restrict(   coefficients,
                        element,
                        cell,
                        cell.get_vertex_coordinates(),
                        cell)

        for particle in pop[cell.index()]:
            element.evaluate_basis_all( basis_matrix,
                                        particle.x,
                                        cell.get_vertex_coordinates(),
                                        cell.orientation())

            phii = np.dot(coefficients, basis_matrix)[:]

            q = particle.q
            PE += 0.5*q*phii

    return PE
