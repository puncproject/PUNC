from __future__ import print_function
from poisson import *
import dolfin as df
import numpy as np

def solve_electric_field(V, facet_f, n_components, outer_Dirichlet_bcs):
    """ This function solves Laplace's equation, $\del^2\varPhi = 0$, for each
    surface component j with boundary condition $\varPhi = 1V$ on component j
    and $\varPhi=0$ on every other component, including the outer boundaries.

    Args:
         V           : FunctionSpace(mesh, "CG", 1)
         W           : VectorFunctionSpace(mesh, 'DG', 0)
         facet_f     : contains the facets of each surface component
         n_components: number of surface components
         outer_Dirichlet_bcs: Dirichlet boundary condition, $\varPhi=0$, at the
                              outer simulation domain

    returns:
            A list of calculated electric fields for every surface componet.
    """
    # Solve Laplace equation for each electrical component
    poisson = PoissonSolverDirichlet(V, outer_Dirichlet_bcs)

    E_object = []
    for i in range(n_components):
        object_bcs = []
        for j in range(n_components):
            # Object boundary value
            # 1 at i = j and 0 at the others
            if i == j:
                c = df.Constant(1.0)
            else:
                c = df.Constant(0.0)
            bc_j = df.DirichletBC(V, c, facet_f, j+1) # facet indexing starts at 1
            object_bcs.append(bc_j)

        # Source term: 0 everywhere
        rho = df.Function(V)
        phi = poisson.solve(rho, object_bcs)
        E = electric_field(phi)
        #plot(phi, interactive=True)
        E_object.append(E)

    return E_object

def capacitance_matrix(V, mesh, facet_f, n_components, epsilon_0):
    """ This function calculates the mutual capacitance matrix, $C_{i,j}$.
    The elements of mutual capacitance matrix are given by:

     C_{i,j} = \epsilon_0\int_{\Omega_i}\mathbf{E}_{j}\cdot\hat{n}_i d\sigma_i.

     For each surface component j, Laplace's equation, $\del^2\varPhi = 0$, is
     solved with boundary condition $\varPhi = 1V$ on component j and
     $\varPhi=0$ on every other component, including the outer boundaries.


    Args:
         V           : FunctionSpace(mesh, "CG", 1)
         W           : VectorFunctionSpace(mesh, 'DG', 0)
         mesh        : the mesh of the simulation domain
         facet_f     : contains the facets of each surface component
         n_components: number of surface components
         epsilon_0   : Permittivity of vacuum

    returns:
            The inverse of the mutual capacitance matrix
    """
    outer_Dirichlet_bcs = dirichlet_bcs(V, facet_f, n_components)
    E_object = solve_electric_field(V, facet_f, n_components,
                                    outer_Dirichlet_bcs)

    ds = df.Measure('ds', domain = mesh, subdomain_data = facet_f)
    n = df.FacetNormal(mesh)
    capacitance = np.empty((n_components, n_components))
    for i in range(n_components):
        for j in range(n_components):
            capacitance[i,j] = epsilon_0*\
                               df.assemble(df.inner(E_object[j], -1*n)*ds(i+1))

    inv_capacitance = np.linalg.inv(capacitance)
    print("                               ")
    print("Mutual capacitance matrix:     ")
    print("                               ")
    print(capacitance)
    print("-------------------------------")
    print("                               ")
    print("Inverse of capacitance matrix: ")
    print("                               ")
    print(inv_capacitance)
    print("-------------------------------")
    print("                               ")

    return inv_capacitance


def circuits(inv_capacitance, circuits_info):
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
    tmp1 = []
    tmp2 = []
    for i in range(len(circuits_info)):
        if len(circuits_info[i]) > 1:
            tmp1.append(len(circuits_info[i]))
            tmp2.append(i)
    D_matrices = []
    for i in range(len(tmp1)):
        D_matrices.append(np.ones((tmp1[i], tmp1[i])))
    for i in range(len(D_matrices)):
        circuit = circuits_info[tmp2[i]]
        n_comp = len(circuit)
        for k in range(1, n_comp):
            for l in range(n_comp):
                D_matrices[i][k-1,l] = \
                                 inv_capacitance[circuit[k]-1, circuit[l]-1] -\
                                 inv_capacitance[circuit[0]-1, circuit[l]-1]

    inv_D_matrices = []
    for i in range(len(D_matrices)):
        inv_D_matrices.append(np.linalg.inv(D_matrices[i]))

    print("                                          ")
    print("Difference capacitance matrices:          ")
    print("                                          ")
    print(D_matrices)
    print("------------------------------------------")
    print("                                          ")
    print("Inverse of difference capacitance matrix: ")
    print("                                          ")
    print(inv_D_matrices)
    print("------------------------------------------")
    print("                                          ")
    return inv_D_matrices

if __name__=='__main__':

    from mark_object import *
    from get_object import *

    dim = 2
    epsilon_0 = 1.0
    n_components = 4
    object_type = 'multi_circles'

    circuits_info = [[1, 3], [2, 4]]

    mesh = df.Mesh("circuit.xml")
    V = FunctionSpace(mesh, "CG", 1)

    d = mesh.geometry().dim()
    L = np.empty(2*d)
    for i in range(d):
        l_min = mesh.coordinates()[:,i].min()
        l_max = mesh.coordinates()[:,i].max()
        L[i] = l_min
        L[d+i] = l_max
    object_info = get_object(dim,
                             object_type,
                             n_components)
    facet_f = mark_boundaries(mesh,
                              L,
                              object_type,
                              object_info,
                              n_components)
    inv_capacitance = capacitance_matrix(V,
                                         mesh,
                                         facet_f,
                                         n_components,
                                         epsilon_0)

    inv_D = circuits(inv_capacitance, circuits_info)
