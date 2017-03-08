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

def kineticEnergy(pop):
    """
    Computes kinetic energy at current velocity time step.
    Useful for the first (zeroth) time step before velocity has between
    advanced half a timestep. To get velocity between two velocity time
    steps (e.g. at integer steps after the start-up) use accel() return.
    """
    KE = 0
    for cell in pop:
        for particle in cell:
            m = particle.m
            v = particle.v
            KE += 0.5*m*np.dot(v,v)
    return KE

def potentialEnergy(pop,phi):

    PE = 0

    V = phi.function_space()
    element = V.dolfin_element()
    sDim = element.space_dimension()  # Number of nodes per element
    basisMatrix = np.zeros((sDim,1))
    coefficients = np.zeros(sDim)

    for cell in df.cells(pop.mesh):
        phi.restrict(   coefficients,
                        element,
                        cell,
                        cell.get_vertex_coordinates(),
                        cell)

        for particle in pop[cell.index()]:
            element.evaluate_basis_all( basisMatrix,
                                        particle.x,
                                        cell.get_vertex_coordinates(),
                                        cell.orientation())

            phii = np.dot(coefficients, basisMatrix)[:]

            q = particle.q
            PE += 0.5*q*phii

    return PE
