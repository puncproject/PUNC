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

from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

from punc.poisson import *
import dolfin as df
import numpy as np

def get_measure(mesh, boundaries):
    return df.Measure("ds", domain=mesh, subdomain_data=boundaries)

def get_facet_normal(mesh):
    return df.FacetNormal(mesh)

class Object(df.DirichletBC):
    """
    An Object is a subdomain of DirichletBC class that represents an electrical
    component.

    Attributes:
        charge (:obj: float) : The total charge of the object.

        interpolated_charge (:obj: float): The total interpolated charge of the
        object.

        _potential (:obj: list): The electric potential of the object.

        dofs (:obj: list): A list of dof indices of the finite element
        function space corresponding to the object.

        inside (:obj: method): Specifies the boundary of the object. Returns
        True if a point is inside the object domain, and False otherwise. It is
        also used to specify on which facets the boundary conditions should be
        applied.

        V (DOLFIN: FunctionSpace): The finite element function space.
    """

    def __init__(self, V, bnd, bnd_id, charge=0, potential=0, floating=True):
        """
        Constructor

        Args:
            V (DOLFIN: FunctionSpace): The finite element function space.

            sub_domain (DOLFIN: DirichletBC): Subdomain of DirichletBC class.

            method (str): Optional argument: A string specifying the method to
            identify dofs.
        """

        self.charge = charge
        self.floating = floating
        self._potential = potential
        self.interpolated_charge = 0
        self.V = V

        potential = df.Constant(potential)

        # Can either take a sub_domain as a SubDomain-inherited class or as a
        # FacetFunciton domain and an id. The latter is more general since
        # boundaries can be arbitrarily specified in gmsh, but it does not
        # provide the inside() function which is still used in old parts of the
        # code. This will be deleted in the future, and the latter version used.
        if bnd==None:
            df.DirichletBC.__init__(self, V, potential, bnd_id, "topological")
            self.inside = self.domain_args[0].inside
        else:
            df.DirichletBC.__init__(self, V, potential, bnd, bnd_id, "topological")
            self.id = bnd_id

        self.dofs = self.get_boundary_values().keys()

    def add_charge(self, q):
        """
        Adds charge, q, to the object charge.

        Args:
             q (float): The electric charge to be added to object.
        """
        self.charge += q

    def compute_interpolated_charge(self, q):
        """
        Computes the interpolated charge to the object.

        Args:
            q (DOLFIN array): The interpolated charge.
        """
        self.interpolated_charge = np.sum(q.vector()[self.dofs])

    def set_potential(self, potential):
        """
        Sets the object potential.

        Args:
            potential (float): The electric potential.
        """
        self._potential = potential
        self.set_value(df.Constant(self._potential))

    def vertices(self):
        """
        Returns the vertices of the surface of the object

        This information might be useful for calculating the current density
        into the object surface
        """
        coords = self.V.mesh().coordinates()
        d2v = df.dof_to_vertex_map(self.V)
        vertex_indices = list(set(d2v[self.dofs]))
        return coords[vertex_indices]

    def cells(self, facet_func, id):
        """
        Returns the cells adjacent to the surface of the object

        This information might be useful for calculating the current density
        into the object surface
        """
        mesh = self.V.mesh()
        D = mesh.topology().dim()
        mesh.init(D-1,D) # Build connectivity between facets and cells
        itr_facet = df.SubsetIterator(facet_func, id)
        object_adjacent_cells = []
        for f in itr_facet:
            object_adjacent_cells.append(f.entities(D)[0])
        return object_adjacent_cells

    def mark_facets(self, facet_func, id):
        """
        Marks the surface facets of the object

        This function is needed for calculating the capacitance matrix
        """
        object_boundary = df.AutoSubDomain(lambda x: self.inside(x, True))
        object_boundary.mark(facet_func, id)
        return facet_func

    def mark_cells(self, cell_f, facet_func, id):
        """
        Marks the cells adjacent to the object

        This information might be useful for calculating the current density
        into the object surface
        """
        cells = self.cells(facet_func, id)
        for c in cells:
            cell_f[int(c)] = id
        return cell_f

def reset_objects(objects):
    """
    Resets the potential for each object.
    """
    for o in objects:
        o.set_potential(df.Constant(0.0))

def compute_object_potentials(objects, E, inv_cap_matrix, mesh, bnd):
    """
    Calculates the image charges for all the objects, and then computes the
    potential for each object by summing over the difference between the
    collected and image charges multiplied by the inverse of capacitance matrix.
    """

    ds = df.Measure("ds", domain=mesh, subdomain_data=bnd)
    normal = df.FacetNormal(mesh)

    image_charge = [None]*len(objects)
    for i, o in enumerate(objects):
        flux = df.inner(E, -1 * normal) * ds(o.id)
        image_charge[i] = df.assemble(flux)

    for i, o in enumerate(objects):
        object_potential = 0.0
        for j, p in enumerate(objects):
             object_potential += (p.charge - image_charge[j])*inv_cap_matrix[i,j]
        o.set_potential(df.Constant(object_potential))

class Circuit(object):
    """
    A circuit is a collection of an arbitrary number of electrical components
    (represented as Object objects), and it is a disjoint entity from the rest
    of the system.


    Attributes:
        objects (:obj: list of Object objets): A list of Object objects that
        together constitute the circuit.

        charge (:obj: float) : The total charge in the circuit

        precomputed_charge (:obj: list): component (Object object) charge
        precomputed from the first part of the inverse of the bias matrix

                inv_bias_mat[circuit_members, :-len(num_circuits)]

        and the bias potential vector, phi_bias:

                precomputed_charge = dot(inv_bias_mat, phi_bias)

        bias_mat (:obj: array): The second part of the inverse of the bias
        matrix:

                bias_mat = inv_bias_mat[circuit_members, -len(num_circuits):]
    """
    def __init__(self, objects, precomputed_charge, bias_mat):
        """
        The constructor

        Args:
            objects (Object objets)  : List of electrical components

            precomputed_charge (list): Precomputed componet charge

            bias_mat (array)         : The second part of the inverse of the
            bias matrix

                bias_mat = inv_bias_mat[circuit_members, -len(num_circuits):]
        """
        self.objects = objects
        self.charge = 0
        self.precomputed_charge = precomputed_charge
        self.bias_mat = bias_mat

    def circuit_charge(self):
        """
        Computes the total charge in the circuit, and sets the value to circuit
        charge: self.charge.
        """
        circuit_charge = 0.0
        for o in self.objects:
            circuit_charge += (o.charge - o.interpolated_charge)
        self.charge = circuit_charge

    def redistribute_charge(self, tot_charge):
        """
        Redistributes the charge in the circuit among the circuits electrical
        components.

        Args:
            tot_charge (list): A list containing the total charge in every
            circuit.
        """
        redistributed_charge = np.dot(self.bias_mat, tot_charge)
        for i, o in enumerate(self.objects):
            o.charge = self.precomputed_charge[i] + \
                          redistributed_charge[i] + \
                                           o.interpolated_charge

def redistribute_circuit_charge(circuits):
    """
    The total charge in each circuit is redistributed based on the predefined
    relative bias potentials between the electrical components in the circuit.
    """
    num_circuits = len(circuits)
    tot_charge = [0]*num_circuits
    for i, c in enumerate(circuits):
        c.circuit_charge()
        tot_charge[i] = c.charge

    for c in circuits:
        c.redistribute_charge(tot_charge)

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


def solve_laplace(V, poisson, objects, boundaries, ext_bnd_id):
    """
    This function solves Laplace's equation, div grad phi = 0, for each
    surface component j with boundary condition phi = 1 on component j
    and phi = 0 on every other component.

    Args:
         V                : DOLFIN function space
         poisson          : Poisson solver
         objects          : A list containing all the objects
         boundaries       : DOLFIN MeshFunction over facet regions
         ext_bnd_id       : The number given to the exterior facet regions
                            in gmsh
    returns:
            A list of calculated electric fields for every surface component
            (object).
    """

    bcs = poisson.bcs
    poisson.bcs = [df.DirichletBC(V, df.Constant(0.0), boundaries, ext_bnd_id)]
    esolver = ESolver(V)
    num_objects = len(objects)
    object_e_field = [0.0] * num_objects
    for i, o in enumerate(objects):
        for j, p in enumerate(objects):
            if i == j:
                p.set_potential(1.0)
            else:
                p.set_potential(0.0)

        rho = df.Function(V)
        phi = poisson.solve(rho, objects)
        object_e_field[i] = esolver.solve(phi)
    poisson.bcs = bcs
    return object_e_field


def capacitance_matrix(V, poisson, objects, boundaries, bnd_id):
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
          objects          : A list containing all the objects
          boundaries       : DOLFIN MeshFunction over facet regions
          bnd_id           : The number given to the exterior facet regions
                             in gmsh
    returns:
            The inverse of the mutual capacitance matrix
    """
    mesh = V.mesh()
    
    num_objects = len(objects)
    capacitance = np.empty((num_objects, num_objects))

    object_e_field = solve_laplace(V, poisson, objects, boundaries, bnd_id)

    ds = df.Measure('ds', domain=mesh, subdomain_data=boundaries)
    n = df.FacetNormal(mesh)

    for i in range(num_objects):
        for j in range(num_objects):
            flux = df.inner(object_e_field[j], -1 * n) * ds(objects[i].id)
            capacitance[i, j] = df.assemble(flux)
 
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

    num_components = inv_cap_matrix.shape[0]  # Total number of components
    num_circuits = len(circuits_info)  # Number of circuits
    s = 0
    for i in range(num_circuits):
        circuit = circuits_info[i]
        bias_matrix[i - num_circuits, circuit] = 1.0
        for j in range(1, len(circuit)):
            for k in range(num_components):
                bias_matrix[j - 1 + s, k] = inv_cap_matrix[circuit[j], k] -\
                    inv_cap_matrix[circuit[0], k]
        s += len(circuit) - 1

    return np.linalg.inv(bias_matrix)
