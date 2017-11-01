from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

from punc.poisson import *
import dolfin as df
import numpy as np

import itertools as itr

def markers(mesh, objects):
    """
    Marks the facets on the boundary of the objects.

    Args:
         mesh   : The mesh of the simulation domain
         objects: A list containing all the objects

    returns:
            The marked facets of the objects.
    """
    num_objects = len(objects)

    facet_func = df.FacetFunction('size_t', mesh)
    facet_func.set_all(num_objects)

    for i, o in enumerate(objects):
        facet_func = o.mark_facets(facet_func, i)

    return facet_func

def solve_laplace(V, poisson, non_periodic_bnd, objects):
    """
    This function solves Laplace's equation, div grad phi = 0, for each
    surface component j with boundary condition phi = 1 on component j
    and phi = 0 on every other component.

    Args:
         V                : DOLFIN function space
         poisson          : Poisson solver
         non_periodic_bnd : Non-periodic boundaries
         objects          : A list containing all the objects

    returns:
            A list of calculated electric fields for every surface component.
    """
    assert not all(bnd for bnd in non_periodic_bnd.periodic), \
    "The system cannot be solved as a periodic boundary value problem."
    
    bcs = poisson.bcs
    poisson.bcs = [df.DirichletBC(V, df.Constant(0.0), non_periodic_bnd)]

    num_objects = len(objects)
    object_e_field = [0.0]*num_objects
    for i, o in enumerate(objects):
        for j, p in enumerate(objects):
            if i == j:
                p.set_potential(1.0)
            else:
                p.set_potential(0.0)

        rho = df.Function(V)
        phi = poisson.solve(rho, objects)
        object_e_field[i] = electric_field(phi)
    poisson.bcs = bcs
    return object_e_field

def capacitance_matrix(V, poisson, non_periodic_bnd, objects):
    """
    This function calculates the mutual capacitance matrix, C_ij.
    The elements of mutual capacitance matrix are given by:

     C_ij = integral_Omega_i inner(E_j, n_i) dsigma_i.

     For each surface component j, Laplace's equation, div grad phi = 0, is
     solved with boundary condition phi = 1 on component j and
     phi = 0 on every other component, including the outer boundaries.


    Args:
          V                : DOLFIN function space
          poisson          : Poisson solver
          non_periodic_bnd : Non-periodic boundaries
          objects          : A list containing all the objects

    returns:
            The inverse of the mutual capacitance matrix
    """
    mesh = V.mesh()

    facet_func = markers(mesh, objects)

    num_objects = len(objects)
    capacitance = np.empty((num_objects, num_objects))

    object_e_field = solve_laplace(V, poisson, non_periodic_bnd, objects)

    ds = df.Measure('ds', domain = mesh, subdomain_data = facet_func)
    n = df.FacetNormal(mesh)

    for i in range(num_objects):
        for j in range(num_objects):
            flux = df.inner(object_e_field[j], -1*n)*ds(i)
            capacitance[i,j] = df.assemble(flux)

    return np.linalg.inv(capacitance)

def bias_matrix(inv_cap_matrix, circuits_info):
    """ This function calculates the matrix $D_{\gamma}$ for potential biases
    between different surface components in each disjoint circuit. Each circuit
    $\gamma$ can contain $n_{\gamma}$ componets indexed $c_{\gamma,l}$ with
    $1\leq l \geq n_{\gamma}$. The matrix $D_{\gamma}$ is given by

    D_{\gamma,i,j} =  C^{-1}_{c_{\gamma,i},c_{\gamma,j}} -
                      C^{-1}_{c_{\gamma,0},c_{\gamma,j}}

    Args:
         inv_capacitance: Inverse of mutual capacitance matrix
         circuits_info  : A list of all disjoint circuits. Each circuit is
                          composed of different surface components, and each
                          surface component is represented by an integer which
                          corresponds to the number given to the facets of that
                          component.

    returns:
            A list of the inverse of the matrix $D_{\gamma}$, for all circuits.
    """

    bias_matrix = np.zeros(inv_cap_matrix.shape)

    num_components = inv_cap_matrix.shape[0] # Total number of components
    num_circuits = len(circuits_info) # Number of circuits
    s = 0
    for i in range(num_circuits):
        circuit = circuits_info[i]
        bias_matrix[i-num_circuits, circuit] = 1.0
        for j in range(1, len(circuit)):
            for k in range(num_components):
                bias_matrix[j-1+s,k] = inv_cap_matrix[circuit[j], k] -\
                                       inv_cap_matrix[circuit[0], k]
        s += len(circuit)-1

    return np.linalg.inv(bias_matrix)
