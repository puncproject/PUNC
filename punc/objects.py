from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np

class VObject(df.DirichletBC):

    def __init__(self, V, potential, sub_domains, sub_domain, method="topological"):
        df.DirichletBC.__init__(self, V, potential, sub_domains, sub_domain, method)
        self.charge = 0
        self._potential = 0
        self._sub_domain = sub_domain

    def set_potential(self, potential):
        self._potential = potential
        self.set_value(df.Constant(self._potential))


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

    def __init__(self, V, sub_domain, sub_domains=None, charge=0, potential=0, method="topological"):
        """
        Constructor

        Args:
            V (DOLFIN: FunctionSpace): The finite element function space.

            sub_domain (DOLFIN: DirichletBC): Subdomain of DirichletBC class.

            method (str): Optional argument: A string specifying the method to
            identify dofs.
        """

        self.charge = charge
        self._potential = potential
        self.interpolated_charge = 0
        self.V = V

        potential = df.Constant(potential)

        # Can either take a sub_domain as a SubDomain-inherited class or as a
        # FacetFunciton domain and an id. The latter is more general since
        # boundaries can be arbitrarily specified in gmsh, but it does not
        # provide the inside() function which is still used in old parts of the
        # code. This will be deleted in the future, and the latter version used.
        if sub_domains==None:
            df.DirichletBC.__init__(self, V, potential, sub_domain, method)
            self.inside = self.domain_args[0].inside
        else:
            df.DirichletBC.__init__(self, V, potential, sub_domains, sub_domain, method)
            self._sub_domain = sub_domain

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
    for o in objects:
        o.set_potential(df.Constant(0.0))

def compute_object_potentials(q, objects, inv_cap_matrix):
    """
    Sets the interpolated charge to the objects, and then computes the object
    potential by using the inverse of capacitance matrix.
    """
    for o in objects:
        o.compute_interpolated_charge(q)

    for i, o in enumerate(objects):
        potential = 0.0
        for j, p in enumerate(objects):
            potential += (p.charge - p.interpolated_charge)*inv_cap_matrix[i,j]
        o.set_potential(potential)

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
