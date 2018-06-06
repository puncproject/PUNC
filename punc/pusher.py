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

def accel(pop, E, dt):

    W = E.function_space()
    mesh = W.mesh()
    element = W.dolfin_element()
    s_dim = element.space_dimension()  # Number of nodes per element
    v_dim = element.value_dimension(0) # Number of values per node (=geom. dim.)
    basis_matrix = np.zeros((s_dim,v_dim))
    coefficients = np.zeros(s_dim)
    dim = mesh.geometry().dim()

    KE = 0.0
    for cell in df.cells(mesh):

        E.restrict( coefficients,
                    element,
                    cell,
                    cell.get_vertex_coordinates(),
                    cell)

        for particle in pop[cell.index()]:
            element.evaluate_basis_all( basis_matrix,
                                        particle.x,
                                        cell.get_vertex_coordinates(),
                                        cell.orientation())

            Ei = np.dot(coefficients, basis_matrix)[:]

            m = particle.m
            q = particle.q

            vel = particle.v

            inc = dt*(q/m)*Ei

            KE += 0.5*m*np.dot(vel,vel+inc) # This has error O(dt^2)

            particle.v += inc


    return KE

def boris(pop, E, B, dt):

    W = E.function_space()
    mesh = W.mesh()
    element = W.dolfin_element()
    s_dim = element.space_dimension()  # Number of nodes per element
    v_dim = element.value_dimension(0) # Number of values per node (=geom. dim.)
    basis_matrix = np.zeros((s_dim,v_dim))
    coefficients = np.zeros(s_dim)
    dim = mesh.geometry().dim()
    assert dim == 3
    KE = 0.0
    for cell in df.cells(mesh):

        E.restrict( coefficients,
                    element,
                    cell,
                    cell.get_vertex_coordinates(),
                    cell)

        for particle in pop[cell.index()]:
            element.evaluate_basis_all( basis_matrix,
                                        particle.x,
                                        cell.get_vertex_coordinates(),
                                        cell.orientation())

            Ei = np.dot(coefficients, basis_matrix)[:]

            m = particle.m
            q = particle.q

            vel = particle.v
            
            t = np.tan((dt*q/(2.*m))*B)
            s = 2.*t/(1.+t[0]**2+t[1]**2+t[2]**2)
            v_minus = vel + 0.5*dt*(q/m)*Ei

            KE += 0.5*m*np.dot(v_minus,v_minus)

            v_minus_cross_t = np.cross(v_minus, t)
            v_prime = v_minus + v_minus_cross_t
            v_prime_cross_s = np.cross(v_prime, s)
            v_plus = v_minus + v_prime_cross_s
            particle.v = v_plus[:] + 0.5*dt*(q/m)*Ei

    return KE


def boris_nonuniform(pop, E, B, dt):

    W = E.function_space()
    mesh = W.mesh()
    element = W.dolfin_element()
    s_dim = element.space_dimension()  # Number of nodes per element
    # Number of values per node (=geom. dim.)
    v_dim = element.value_dimension(0)
    basis_matrix = np.zeros((s_dim, v_dim))
    coefficients = np.zeros(s_dim)
    mag_coefficients = np.zeros(s_dim)
    dim = mesh.geometry().dim()
    assert dim == 3
    KE = 0.0
    for cell in df.cells(mesh):

        E.restrict(coefficients,
                   element,
                   cell,
                   cell.get_vertex_coordinates(),
                   cell)

        B.restrict(mag_coefficients,
                   element,
                   cell,
                   cell.get_vertex_coordinates(),
                   cell)

        for particle in pop[cell.index()]:
            element.evaluate_basis_all(basis_matrix,
                                       particle.x,
                                       cell.get_vertex_coordinates(),
                                       cell.orientation())

            Ei = np.dot(coefficients, basis_matrix)[:]
            Bi = np.dot(mag_coefficients, basis_matrix)[:]

            m = particle.m
            q = particle.q

            vel = particle.v
            
            t = np.tan((dt * q / (2. * m)) * Bi)
            s = 2. * t / (1. + t[0]**2 + t[1]**2 + t[2]**2)
            v_minus = vel + 0.5 * dt * (q / m) * Ei

            KE += 0.5 * m * np.dot(v_minus, v_minus)

            v_minus_cross_t = np.cross(v_minus, t)
            v_prime = v_minus + v_minus_cross_t
            v_prime_cross_s = np.cross(v_prime, s)
            v_plus = v_minus + v_prime_cross_s
            particle.v = v_plus[:] + 0.5 * dt * (q / m) * Ei

    return KE

def move_periodic(pop, Ld, dt):

    for cell in pop:
        for particle in cell:
            particle.x += dt*particle.v
            particle.x %= Ld

def move(pop, dt):

    for cell in pop:
        for particle in cell:
            particle.x += dt*particle.v
