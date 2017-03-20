from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

from poisson import *
import dolfin as df
import numpy as np
from object import *
import itertools as itr

def markers(mesh, Ld, objects):

    dim = len(Ld)
    L = np.zeros(2*dim)
    L[dim:] = Ld
    n_components = len(objects)

    facet_f = df.FacetFunction('size_t', mesh)
    facet_f.set_all(n_components+2*dim)

    for i, o in enumerate(objects):
        facet_f = o.mark_facets(facet_f, i)

    for i in range(2*dim):
        boundary = 'near((x[i]-l), 0, tol)'
        boundary = df.CompiledSubDomain(boundary, i=i%dim, l=L[i], tol = 1E-8)
        boundary.mark(facet_f, (n_components + i))

    return facet_f

def exterior_boundaries(V, facet_f, n_components):
    dim = V.mesh().geometry().dim()
    boundaries = [0]*2*dim
    for i in range(2*dim):
        boundaries[i] = df.DirichletBC(V, df.Constant(0.0), facet_f, n_components + i)

    return boundaries

def solve_laplace(V, exterior_bcs, objects):
    """
    This function solves Laplace's equation, $\del^2\varPhi = 0$, for each
    surface component j with boundary condition $\varPhi = 1V$ on component j
    and $\varPhi=0$ on every other component, including the outer boundaries.

    Args:
         V           : FunctionSpace(mesh, "CG", 1)
         exterior_bcs: Dirichlet boundary conditions on exterior boundaries
         objects     : A list containing all the objects

    returns:
            A list of calculated electric fields for every surface componet.
    """
    n_components = len(objects)
    poisson = PoissonSolver(V, exterior_bcs)

    object_e_field = [0.0]*n_components
    for i, o in enumerate(objects):
        for j, p in enumerate(objects):
            if i == j:
                p.set_potential(1.0)
            else:
                p.set_potential(0.0)

        rho = df.Function(V)
        phi = poisson.solve(rho, objects)
        object_e_field[i] = electric_field(phi)

    return object_e_field

def capacitance_matrix(mesh, Ld, circle):
    """
    This function calculates the mutual capacitance matrix, $C_{i,j}$.
    The elements of mutual capacitance matrix are given by:

     C_{i,j} = \epsilon_0\int_{\Omega_i}\mathbf{E}_{j}\cdot\hat{n}_i d\sigma_i.

     For each surface component j, Laplace's equation, $\del^2\varPhi = 0$, is
     solved with boundary condition $\varPhi = 1V$ on component j and
     $\varPhi=0$ on every other component, including the outer boundaries.


    Args:
         mesh              : the mesh of the simulation domain
         Ld                : the size of the simulation domain
         objects_boundaries: a list of objects represented by DirichletBC

    returns:
            The inverse of the mutual capacitance matrix
    """
    V = df.FunctionSpace(mesh, "CG", 1)
    objects = circle.get_objects(V)
    # objects = [None]*len(objects_boundaries)
    # for i, c in enumerate(objects_boundaries):
    #     objects[i] = Object(V, c)

    facet_f = markers(mesh, Ld, objects)

    exterior_bcs = exterior_boundaries(V, facet_f, len(objects))

    n_components = len(objects)
    capacitance = np.empty((n_components, n_components))

    object_e_field = solve_laplace(V, exterior_bcs, objects)

    ds = df.Measure('ds', domain = mesh, subdomain_data = facet_f)
    n = df.FacetNormal(mesh)

    for i in range(n_components):
        for j in range(n_components):
            capacitance[i,j] = \
                            df.assemble(df.inner(object_e_field[j], -1*n)*ds(i))

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

def init_circuits(objects, inv_cap_matrix, circuits_info, bias_potential):

    num_circuits = len(circuits_info)
    inv_bias_matrix = bias_matrix(inv_cap_matrix, circuits_info)

    bias_potential = list(itr.chain(*bias_potential))

    bias_0 = np.dot(inv_bias_matrix[:,:len(bias_potential)], bias_potential)

    circuits = []
    for i in range(num_circuits):
        circuit_comps = []
        circuit = circuits_info[i]
        for j in circuit:
            circuit_comps.append(objects[j])
            a = np.array([np.pi, np.pi])

        circuits.append(Circuit(circuit_comps, bias_0[circuit],\
                        inv_bias_matrix[circuit,len(bias_potential):]))

    return circuits
