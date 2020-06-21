# Copyright (C) 2017, Sigvald Marholm, Diako Darian and Mikael Mortensen
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
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PUNC. If not, see <http://www.gnu.org/licenses/>.

import dolfin as df
import numpy as np
import copy
import os

class ConstantBC(df.DirichletBC):
    """
    Enforces a constant but unknown boundary. The (single) unknown value at the
    boundary must be determined from the variational formaulation, typically by
    means of a Lagrange multiplier. See examples in the demos.

    Tested for 1st and 2nd order Lagrange finite elements but should in principe
    work for higher orders as well.

    This class is in most ways similar to Dolfin's own DirichletBC class, which
    it inherits. Its constructor is similar to DirichletBC's except that the
    second argument (the value on the boundary) must be omitted, e.g.:

        bc = ConstantBC(V, sub_domain)
        bc = ConstantBC(V, sub_domain, method)
        bc = ConstantBC(V, sub_domains, sub_domain)
        bc = ConstantBC(V, sub_domains, sub_domain, method)

    where sub_domain, sub_domains and method has the same meanings as for
    DirichletBC.
    """

    def __init__(self, *args, **kwargs):

        # Adds the missing argument (the value on the boundary) before calling
        # the parent constructor. The value must be zero to set the
        # corresponding elements in the load vector to zero.

        args = list(args)
        args.insert(1, df.Constant(0.0))
        self.monitor = False
        self.compiled_apply = kwargs.pop('compiled_apply', True)
        if self.compiled_apply:
            thisdir = os.path.dirname(__file__)
            path = os.path.join(thisdir, 'apply.cpp')
            code = open(path, 'r').read()
            self.compiled_apply = df.compile_cpp_code(code)

        df.DirichletBC.__init__(self, *args, **kwargs)

    def apply(self, *args):

        for A in args:

            if isinstance(A, df.GenericVector):
                # Applying to load vector.
                # Set all elements to zero but leave the first.

                ind = self.get_boundary_values().keys()
                first_ind = list(ind)[0]
                first_element = A[first_ind]

                df.DirichletBC.apply(self, A)

                A[first_ind] = first_element

            else:
                # Applying to stiffness matrix.
                # Leave the first row on the boundary node, but change the
                # remaining to be the average of it's neighbors also on the
                # boundary.

                index_dtype = df.la_index_dtype()

                ind = self.get_boundary_values().keys()
                if self.compiled_apply:
                    self.compiled_apply.apply(A, np.array(list(ind), dtype=index_dtype))

                else:
                    length = len(list(ind))-2
                    allneighbors = []
                    inda = np.array(list(ind), dtype=index_dtype)
                    for it, i in enumerate(inda[1:]):
                        allneighbors.append(A.getrow(i)[0])
                    zero_rows = np.array(inda[1:], dtype=index_dtype)
                    A.zero(zero_rows)

                    for it, i in enumerate(inda[1:]):
                        if self.monitor:
                            print("ConstantBC iteration", it, "of", length)
                        neighbors = allneighbors[it]
                        surface_neighbors = np.array([n for n in neighbors if n in ind])
                        values = -np.ones(surface_neighbors.shape)
                        self_index = np.where(surface_neighbors==i)[0][0]
                        num_of_neighbors = len(surface_neighbors)-1
                        values[self_index] = num_of_neighbors
                        A.setrow(i, surface_neighbors, values)
                    A.apply('insert')

    def get_free_row(self):
        bnd_rows = self.get_boundary_values().keys()
        first_bnd_row = list(bnd_rows)[0]
        return first_bnd_row

    def get_boundary_value(self, phi):
        return phi.vector()[self.get_free_row()]

class ObjectBC(ConstantBC):

    def __init__(self, V, bnd, bnd_id):

        ConstantBC.__init__(self, V, bnd, bnd_id)

        self.charge = 0.
        self.collected_current = 0.
        self.potential = 0.
        self.id = bnd_id
        mesh = self.function_space().mesh()
        self.n = df.FacetNormal(mesh)
        self.dss = df.Measure("ds", domain=mesh, subdomain_data=bnd)

    def update_charge(self, phi):
        bnd_id = self.domain_args[1]
        projection = df.dot(df.grad(phi), self.n) * self.dss(bnd_id)
        self.charge = df.assemble(projection)
        return self.charge

    def update_potential(self, phi):
        self.potential = self.get_boundary_value(phi)
        return self.potential

    def update(self, phi):
        self.update_charge(phi)
        self.update_potential(phi)
        return self.charge, self.potential

def relabel_bnd(bnd):
    """
    Relabels MeshFunction bnd such that boundaries are marked 1, 2, 3, etc.
    instead of arbitrary numbers. The order is preserved, and by convention the
    first boundary is the exterior boundary. The objects start at 2. The
    background (not marked) is 0.
    """
    new_bnd = bnd
    new_bnd = df.MeshFunction("size_t", bnd.mesh(), bnd.dim())
    new_bnd.set_all(0)

    old_ids = np.array([int(tag) for tag in set(bnd.array())])
    old_ids = np.sort(old_ids)[1:]
    for new_id, old_id in enumerate(old_ids, 1):
        new_bnd.array()[bnd.where_equal(old_id)] = int(new_id)

    num_objects = len(old_ids)-1
    return new_bnd, num_objects

# deprecated
def load_mesh_CB(fname):
    mesh = df.Mesh(fname+".xml")
    bnd  = df.MeshFunction("size_t", mesh, fname+"_facet_region.xml")
    bnd, num_objects = relabel_bnd(bnd)
    return mesh, bnd, num_objects

def get_charge_sharing_set(vsources, node, group):
    # Used by get_charge_sharing_sets()

    group.append(node)

    i = 0
    while i < len(vsources):
        vsource = vsources[i]
        if vsource[0] == node:
            vsources.pop(i)
            get_charge_sharing_set(vsources, vsource[1], group)
        elif vsource[1] == node:
            vsources.pop(i)
            get_charge_sharing_set(vsources, vsource[0], group)
        else:
            i += 1

def get_charge_sharing_sets(vsources, num_objects):
    """
    Given a list of vsources, this will track which sources are charge-sharing.
    Each tuple in the vsources list represents a voltage source. The first two
    numbers are, respectively, the objects connected to the negative and
    positive terminals of the voltage source. The third number is the voltage.
    The object are labelled 0,1,2,... in the same order as in int_bnd_ids. -1
    means system ground.

    Example::

        vsources = [(1,2,1.0),
                    (2,3,2.0),
                    (4,5,3.0),
                    (7,4,2.0),
                    (9,-1,7.),
                    (10,9,2.)]

        get_charge_sharing_sets(vsources, 11)

    Returns::

        [[1, 2, 3], [4, 5, 7], [0], [6], [8]]

    Objects 1,2,3 are connected by the two upper voltage sources, and 4,5,7 by
    the next three. Objects 0, 6 and 8 are not connected to any. Objects 9 and
    10 are grounded.
    """

    vsources = copy.deepcopy(vsources)
    nodes = set(range(num_objects))

    groups = []
    while vsources != []:
        group = []
        get_charge_sharing_set(vsources, vsources[0][0], group)
        groups.append(group)

    for group in groups:
        for node in group:
            if node != -1:
                nodes.remove(node)

    groups = list(filter(lambda group: -1 not in group, groups))

    for node in nodes:
        groups.append([node])

    return groups

class Circuit(object):

    def __init__(self, V, bnd, objects, vsources=None, isources=None,
                 dt=None, int_bnd_ids=None, eps0=1):

        num_objects = len(objects)

        if int_bnd_ids == None:
            int_bnd_ids = [objects[i].domain_args[1] for i in range(num_objects)]

        if vsources == None:
            vsources = []

        if isources == None:
            isources = []
            self.dt = 1
        else:
            assert dt != None
            self.dt = dt

        self.int_bnd_ids = int_bnd_ids
        self.vsources = vsources
        self.isources = isources
        self.objects = objects
        self.eps0 = eps0

        self.groups = get_charge_sharing_sets(vsources, num_objects)

        self.V = V
        mesh = V.mesh()
        R  = df.FunctionSpace(mesh, "Real", 0)
        self.mu = df.TestFunction(R)
        self.phi = df.TrialFunction(V)
        self.dss = df.Measure("ds", domain=mesh, subdomain_data=bnd)
        self.n = df.FacetNormal(mesh)

        thisdir = os.path.dirname(__file__)
        path = os.path.join(thisdir, 'addrow.cpp')
        code = open(path, 'r').read()
        self.compiled = df.compile_cpp_code(code)

        # Rows in which to store charge and potential constraints
        rows_charge    = [g[0] for g in self.groups]
        rows_potential = list(set(range(num_objects))-set(rows_charge))
        self.rows_charge    = [objects[i].get_free_row() for i in rows_charge]
        self.rows_potential = [objects[i].get_free_row() for i in rows_potential]

    def apply(self, *args):
        # NB: Does not modify matrix in-place.
        # Return value must be used, e.g:
        # A, b = circuit.apply(A, b)

        args = list(args)
        for i in range(len(args)):
            if isinstance(args[i], df.GenericVector):
                self.apply_isources_to_object()
                args[i] = self.apply_vsources_to_vector(args[i])
            else:
                args[i] = self.apply_vsources_to_matrix(args[i])

        return args

    def apply_vsources_to_matrix(self, A):
        # NB: Does not modify matrix in-place.
        # Return value must be used, e.g:
        # A = circuit.apply_vsources_to_matrix(A, b)

        # Charge constraints
        for group, row in zip(self.groups, self.rows_charge):
            ds_group = np.sum([self.dss(self.int_bnd_ids[i]) for i in group])
            # ds_group = np.sum([self.dss(self.int_bnd_ids[i], degree=1) for i in group])
            a0 = self.eps0*df.inner(self.mu, df.dot(df.grad(self.phi), self.n))*ds_group
            A0 = df.assemble(a0)
            cols, vals = A0.getrow(0)

            B = df.Matrix()
            self.compiled.addrow(A, B, cols, vals, row, self.V._cpp_object)
            A = B

        # Potential constraints
        for vsource, row in zip(self.vsources, self.rows_potential):
            obj_a_id = vsource[0]
            obj_b_id = vsource[1]

            cols = []
            vals = []

            if obj_a_id != -1:
                dof_a = self.objects[obj_a_id].get_free_row()
                cols.append(dof_a)
                vals.append(-1.0)

            if obj_b_id != -1:
                dof_b = self.objects[obj_b_id].get_free_row()
                cols.append(dof_b)
                vals.append(+1.0)

            cols = np.array(cols, dtype=np.uintp)
            vals = np.array(vals)

            B = df.Matrix()
            self.compiled.addrow(A, B, cols, vals, row, self.V._cpp_object)
            A = B

        return A

    def apply_vsources_to_vector(self, b):

        # Charge constraints
        for group, row in zip(self.groups, self.rows_charge):
            b[row] = np.sum([self.objects[i].charge for i in group])

        # Potential constraints
        for vsource, row in zip(self.vsources, self.rows_potential):
            b[row] = vsource[2]

        return b

    def apply_isources_to_object(self):

        for isource in self.isources:

            if isinstance(isource, ISource):
                obj_a_id = isource.obj_a_id
                obj_b_id = isource.obj_b_id
                obj_a = self.objects[obj_a_id]
                obj_b = self.objects[obj_b_id]

                # TBD: does V have the wrong sign?
                V = obj_a.potential - obj_b.potential # Voltage at step n
                I = isource.get_current(V) # Current at step n+0.5
                dQ = I*self.dt

            else:
                obj_a_id = isource[0]
                obj_b_id = isource[1]
                dQ = isource[2]*self.dt

            if obj_a_id != -1:
                self.objects[obj_a_id].charge -= dQ

            if obj_b_id != -1:
                self.objects[obj_b_id].charge += dQ

class ISource(object):
    def __init__(self, obj_a_id, obj_b_id, I=None):
        self.obj_a_id = obj_a_id
        self.obj_b_id = obj_b_id
        self.I = I

    def get_current(self, Vn):
        return self.I

class RLC(ISource):
    def __init__(self, obj_a_id, obj_b_id, dt, R, L, C, V=None, I=None):
        super().__init__(obj_a_id, obj_b_id)

        self.R = R
        self.L = L
        self.C = C
        self.dt = dt
        self.tau1 = 0.5*dt*R/L
        self.tau2 = 0.5*dt**2/(L*C)

        if V is None:
            # Voltages at time steps n and n-1
            self.V = np.array([0., 0.])
        else:
            self.V = V

        if I is None:
            # Currents at time steps n+0.5, n-0.5 and n-1.5
            self.I = np.array([0., 0., 0.])
        else:
            self.I = I

    def get_current(self, Vn):
        """
        Takes voltage at step n and returns current at step n+0.5.
        Updates the internal state of RLC.
        """
        self.I[1:0] = self.I[2:1]
        self.V[0] = self.V[1]
        self.V[1] = Vn
        self.I[2] = 2*self.I[1] - (1-self.tau1+self.tau2)*self.I[0] + (self.V[1]-self.V[0])*self.dt/self.L
        self.I[2] /= (1+self.tau1+self.tau2)
        return self.I[2]

class ConstantBoundary(df.SubDomain):
    """
    Enforces constant values for both `TrialFunction` and `TestFunction` on a
    boundary with id `bnd_id` as given by the `FacetFunction` named `bnd`.
    Assumes some sort of elements where the vertices on the boundary are nodes,
    but not necessarily the only nodes. E.g. CG1, CG2, ... and so forth.

    Usage::
        mesh = Mesh("mesh.xml")
        bnd  = MeshFunction('size_t', mesh, "mesh_facet_region.xml")
        cb   = ConstantBoundary(mesh, bnd, bnd_id)
        V    = FunctionSpace(mesh, 'CG', 2, constrained_domain=cb)

    Since FEniCS's constrained_domain is limited to analytic expressions it is
    not really suitable to real-world problems with CAD-based meshes.
    ConstantBoundary overcomes this by creating a piecewise linear function on
    the boundary from an arbitrary mesh. Unfortunately, this process is slow. Use
    ConstantBC if possible, as it is more CAD-friendly.
    """

    def __init__(self, mesh, bnd, bnd_id, tol=df.DOLFIN_EPS):

        df.SubDomain.__init__(self)
        self.mesh   = mesh
        self.bnd    = bnd
        self.bnd_id = bnd_id
        self.tol    = tol

        # Pick a random vertex on the bnd (assuming this vertex is a node)
        facet_id    = bnd.where_equal(bnd_id)[0]
        facet       = list(facets(mesh))[facet_id]
        vertex_id   = facet.entities(0)[0]
        self.vertex = mesh.coordinates()[vertex_id]

        self.bnd_facets = bnd.where_equal(bnd_id)

        self.bmesh = df.BoundaryMesh(mesh, 'exterior')
        facet_dim = self.bmesh.topology().dim()
        self.cell_map = self.bmesh.entity_map(facet_dim)

    def on_bnd_id(self, x):

        # If changing this function, keep in mind that it should work for all
        # points on boundary facets. Not just the vertices.

        for i, facet in enumerate(df.cells(self.bmesh)):
            if self.cell_map[i] in self.bnd_facets:
                if facet.distance(df.Point(x)) < self.tol:
                    return True

        return False

    def inside(self, x, on_bnd):
        # Some FEniCS functions (not all) will pass 3D x even in 2D problems
        x = x[:self.mesh.geometry().dim()]
        return np.linalg.norm(x-self.vertex) < self.tol

    def map(self, x, y):
        if self.on_bnd_id(x):
            y[:] = self.vertex
        else:
            y[:] = x[:]
