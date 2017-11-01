# __authors__ = ('Sigvald Marholm <sigvaldm@fys.uio.no>')
# __date__ = '2017-02-22'
# __copyright__ = 'Copyright (C) 2017' + __authors__
# __license__  = 'GNU Lesser GPL version 3 or any later version'

# Imports important python 3 behaviour to ensure correct operation and
# performance in python 2
from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

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

            KE += 0.5*m*np.dot(vel,vel+inc)

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
    mag_coefficients = np.zeros(s_dim)
    dim = mesh.geometry().dim()

    KE = 0.0
    for cell in df.cells(mesh):

        E.restrict( coefficients,
                    element,
                    cell,
                    cell.get_vertex_coordinates(),
                    cell)

        B.restrict( mag_coefficients,
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
            Bi = np.dot(mag_coefficients, basis_matrix)[:]

            m = particle.m
            q = particle.q

            vel = particle.v
            assert dim == 3
            t = np.tan((dt*q/(2.*m))*Bi)
            s = 2.*t/(1.+t[0]**2+t[1]**2+t[2]**2)
            v_minus = vel + 0.5*dt*(q/m)*Ei

            KE += 0.5*m*np.dot(v_minus,v_minus)

            v_minus_cross_t = np.cross(v_minus, t)
            v_prime = v_minus + v_minus_cross_t
            v_prime_cross_s = np.cross(v_prime, s)
            v_plus = v_minus + v_prime_cross_s
            particle.v = v_plus[:] + 0.5*dt*(q/m)*Ei

    return KE

def move_periodic(pop, Ld, dt):

    for cell in pop:
        for particle in cell:
            particle.x += dt*particle.v
            particle.x %= Ld

def move(pop, Ld, dt):

    for cell in pop:
        for particle in cell:
            particle.x += dt*particle.v
