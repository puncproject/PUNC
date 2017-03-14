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
    sDim = element.space_dimension()  # Number of nodes per element
    vDim = element.value_dimension(0) # Number of values per node (=geom. dim.)
    basisMatrix = np.zeros((sDim,vDim))
    coefficients = np.zeros(sDim)
    dim = mesh.geometry().dim()

    KE = 0.0
    for cell in df.cells(mesh):

        E.restrict( coefficients,
                    element,
                    cell,
                    cell.get_vertex_coordinates(),
                    cell)

        for particle in pop[cell.index()]:
            element.evaluate_basis_all( basisMatrix,
                                        particle.x,
                                        cell.get_vertex_coordinates(),
                                        cell.orientation())

            Ei = np.dot(coefficients, basisMatrix)[:]

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
    sDim = element.space_dimension()  # Number of nodes per element
    vDim = element.value_dimension(0) # Number of values per node (=geom. dim.)
    basisMatrix = np.zeros((sDim,vDim))
    coefficients = np.zeros(sDim)
    mag_coefficients = np.zeros(sDim)
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
            element.evaluate_basis_all( basisMatrix,
                                        particle.x,
                                        cell.get_vertex_coordinates(),
                                        cell.orientation())

            Ei = np.dot(coefficients, basisMatrix)[:]
            Bi = np.dot(mag_coefficients, basisMatrix)[:]

            m = particle.m
            q = particle.q

            vel = particle.v
            assert dim == 3
            t = np.tan((dt*q/(2.*m))*Bi)
            s = 2.*t/(1.+t[0]**2+t[1]**2+t[2]**2)
            v_minus = vel + 0.5*dt*(q/m)*Ei
            v_minus_cross_t = np.cross(v_minus, t)
            v_prime = v_minus + v_minus_cross_t
            v_prime_cross_s = np.cross(v_prime, s)
            v_plus = v_minus + v_prime_cross_s
            inc = v_plus[:] + 0.5*dt*(q/m)*Ei

            KE += 0.5*m*np.dot(vel,vel+inc)

            particle.v = inc

    return KE

def movePeriodic(pop, Ld, dt):

    for cell in pop:
        for particle in cell:
            particle.x += dt*particle.v
            particle.x %= Ld
    pop.relocate()

def move(pop, Ld, dt):

    for cell in pop:
        for particle in cell:
            particle.x += dt*particle.v
    pop.relocate()
