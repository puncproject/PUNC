from __future__ import print_function
import dolfin as df
import numpy as np
import itertools as itr

class Object(df.DirichletBC):

    def __init__(self, V, sub_domain, method="topological"):
        df.DirichletBC.__init__(self, V, df.Constant(0), sub_domain, method)
        self.charge = 0
        self.q_rho = 0 # Should preferrably have a better name
        self._potential = 0
        self.dofs = self.get_boundary_values().keys()
        self.inside = self.domain_args[0].inside
        self.V = V

    def add_charge(self, q):
        self.charge += q

    def set_q_rho(self, q):
        self.q_rho = np.sum(q.vector()[self.dofs])

    def set_potential(self, potential):
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

    def cells(self, facet_f, id):
        """
        Returns the cells adjecent to the surface of the object

        This information might be useful for calculating the current density
        into the object surface
        """
        mesh = self.V.mesh()
        D = mesh.topology().dim()
        mesh.init(D-1,D) # Build connectivity between facets and cells
        itr_facet = df.SubsetIterator(facet_f, id)
        object_adjacent_cells = []
        for f in itr_facet:
            object_adjacent_cells.append(f.entities(D)[0])
        return object_adjacent_cells

    def mark_facets(self, facet_f, id):
        """
        Marks the surface facets of the object

        This function is needed for calculating the capacitance matrix
        """
        object_boundary = df.AutoSubDomain(lambda x: self.inside(x, True))
        object_boundary.mark(facet_f, id)
        return facet_f

    def mark_cells(self, cell_f, facet_f, id):
        """
        Marks the cells adjecent to the object

        This information might be useful for calculating the current density
        into the object surface
        """
        cells = self.cells(facet_f, id)
        for c in cells:
            cell_f[int(c)] = id
        return cell_f

def compute_object_potentials(q, objects, inv_cap_matrix):
    """
    This function sets the interpolated charge to the objects, and then
    calculates the object potential by using the inverse of capacitance matrix.
    """
    for o in objects:
        o.set_q_rho(q)

    for i, o in enumerate(objects):
        potential = 0.0
        for j, p in enumerate(objects):
            potential += (p.charge - p.q_rho)*inv_cap_matrix[i,j]
        o.set_potential(potential)

class Circuit(object):

    def __init__(self, objects, bias_0, bias_1):

        self.objects = objects
        self.bias_0 = bias_0
        self.bias_1 = bias_1
        self.charge = 0

    def circuit_charge(self):
        circuit_charge = 0.0
        for o in self.objects:
            circuit_charge += (o.charge - o.q_rho)
        self.charge = circuit_charge

    def redistribute_charge(self, bias):
        bias = np.dot(self.bias_1, bias)
        for i, o in enumerate(self.objects):
            o.charge = self.bias_0[i] + bias[i] + o.q_rho

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
