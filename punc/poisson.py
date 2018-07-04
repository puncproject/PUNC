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

import dolfin as df
import numpy as np

def load_mesh(fname):
    mesh   = df.Mesh(fname+".xml")
    boundaries = df.MeshFunction("size_t", mesh, fname+"_facet_region.xml")
    return mesh, boundaries

def load_h5_mesh(fname):
    mesh = df.Mesh()
    comm = mesh.mpi_comm()
    hdf = df.HDF5File(comm, fname, "r")
    hdf.read(mesh, "/mesh", False)
    subdomains = df.CellFunction("size_t", mesh)
    hdf.read(subdomains, "/subdomains")
    boundaries = df.FacetFunction("size_t", mesh)
    hdf.read(boundaries, "/boundaries")
    return mesh, boundaries, comm

def get_mesh_ids(boundaries, comm=None):
    tags = np.array([int(tag) for tag in set(boundaries.array())])
    tags = np.sort(tags)

    if comm is None:
        tags = tags.tolist()
        return tags[1], tags[2:]
    else:
        comm_mpi4py = comm.tompi4py()
        ids = []
        ids = sum(comm_mpi4py.allgather(tags.tolist()), [])
        ids = sorted(set(ids))
        return ids[1], ids[2:]

def unit_mesh(N, ext_bnd_id=1):
    """
    Given a list of cell divisions, N, returns a mesh with unit size in each
    spatial direction.
    """
    d = len(N)
    mesh_types = [df.UnitIntervalMesh, df.UnitSquareMesh, df.UnitCubeMesh]
    mesh = mesh_types[d-1](*N)
    facet_func = df.FacetFunction('size_t', mesh)
    facet_func.set_all(0)

    class ExtBnd(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    bnd = ExtBnd()
    bnd.mark(facet_func, ext_bnd_id)
    return mesh, facet_func

def simple_mesh(Ld, N, ext_bnd_id=1):
    """
    Returns a mesh for a given list, Ld, containing the size of domain in each
    spatial direction, and the corresponding number of cell divisions N.
    """
    d = len(N)
    mesh_types = [df.RectangleMesh, df.BoxMesh]

    if d==1:
        mesh = df.IntervalMesh(N[0],0.0,Ld[0])
    else:
        mesh = mesh_types[d-2](df.Point(*np.zeros(len(Ld))), df.Point(*Ld), *N)

    facet_func = df.FacetFunction('size_t', mesh)
    facet_func.set_all(0)

    class ExtBnd(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    bnd = ExtBnd()
    bnd.mark(facet_func, ext_bnd_id)
    return mesh, facet_func

def get_mesh_size(mesh):
	"""
	Returns a vector containing the size of the mesh presuming the mesh is
	rectangular and starts in the origin.
	"""
	return np.max(mesh.coordinates(),0)

class PeriodicBoundary(df.SubDomain):
    """
    Defines periodic exterior boundaries of a hypercube as a domain constraint
    suitable for use in DOLFIN's FunctionSpace class. For instance if Ld is a
    numpy array or list specifying the length of the domain:

        constr = PeriodicBoundary(Ld)
        V = FunctionSpace(mesh, 'CG', 1, constrained_domain=constr)

    The domain is expected to start in the origin. The parameter 'periodic' is
    used if only some axes are period, e.g. if only the y-direction is periodic:

        periodic = [False,True,False]
        constr = PeriodicBoundary(Ld,periodic)
        V = FunctionSpace(mesh, 'CG', 1, constrained_domain=constr)
    """

    def __init__(self, Ld, periodic=True):
        df.SubDomain.__init__(self)
        self.Ld = Ld

        if isinstance(periodic,bool):
            periodic = [periodic]*3

        self.periodic = periodic

    # Target domain
    def inside(self, x, on_bnd):
        return bool(on_bnd
            and     any([(df.near(a,0) and p) for a,p in zip(x,self.periodic)]) # On any periodic lower bound
            and not any([df.near(a,b) for a,b in zip(x,self.Ld)]))              # But not any upper bound


    # Map upper periodic edges to lower edges
    def map(self, x, y):
        y[:] = [a-b if (df.near(a,b) and p) else a for a,b,p in zip(x,self.Ld,self.periodic)]

class NonPeriodicBoundary(df.SubDomain):
    """
    Defines non-periodic exterior boundaries of a hypercube as a subdomain
    suitable for use in DOLFIN's DirichletBC class. The parameter 'periodic'
    specifies which axes are periodic, and which are to be used for instance in
    Dirichlet boundaries. The domain is expected to start in the origin.
    Suitable to use together with PeriodicBoundary. E.g. if only the y-axis is
    periodic and the other walls are Dirichlet boundaries:

        periodic = [False,True,False]
        constr = PeriodicBoundary(Ld,periodic)
        bnd = NonPeriodicBoundary(Ld,periodic)

        V = FunctionSpace(mesh, 'CG', 1, constrained_domain=constr)
        bc = DirichletBC(V, Constant(0), bnd)
    """

    def __init__(self, Ld, periodic=False):
        df.SubDomain.__init__(self)
        self.Ld = Ld

        if isinstance(periodic,bool):
            periodic = [periodic]*3

        self.periodic = periodic

    def inside(self, x, on_bnd):
        return bool(on_bnd and (
		      np.any([(df.near(a,0) and not p) for a,p in zip(x,self.periodic)]) or     # On non-periodic lower bound
		      np.any([(df.near(a,b) and not p) for a,b,p in zip(x,self.Ld,self.periodic)]))) # Or non-periodic upper bound

def phi_boundary(B, v_drift):
    """
    Returns a consistent DOLFIN expression of the potential phi for Dirichlet
    boundaries when the magnetic field is B and the drift velocity v_drift.
    v_drift is a vector of

    Arguments:
        B       : Magnetic field vector. Should always be length 3 (even for 2D
                  and 1D domains) or a scalar. If it's scalar it's taken to be
                  in the z-direction.
        v_drift : Drift velocity vector. Should have as many dimesnions as the
                  domain.

    Example:

        bnd = NonPeriodicBoundary(Ld)
        phi0 = phi_boundary(B, v_drift)
        bc = DirichletBC(V, phi0, bnd)
    """

    d = len(v_drift)

    if np.isscalar(B):
        B = [0,0,B]

    assert len(B)==3

    E = -np.cross(v_drift,B)

    phi = 'x[0]*Ex + x[1]*Ey + x[2]*Ez' # DOLFIN sets unused dimesnions to zero
    return df.Expression(phi, degree=1, Ex=E[0], Ey=E[1], Ez=E[2])

class PoissonSolver(object):
    """
    Solves the Poisson Equation on the function space V:

        div grad phi = rho

    Example:

        solver = PoissonSolver(V, bcs_stationary)
        phi = solver.solve(rho, bcs)

    solve() can be called multiple times while the stiffness matrix is
    assembled only in the constructor to save computations.

    Boundary conditions (bcs) can be applied either in solve() and/or in the
    constructor if a boundary should always have the same condition. Overriding
    boundaries in the constructor with boundaries in solve() is not tested but
    may work. bcs can be a single DirichletBC object or an Object object or an
    ObjectBC object or list of such.

    The objects argument do the exact same thing as the bcs argument, but it
    may be convenient to separate e.g. object boundaries and exterior
    boundaries. circuit applies a Circuit object to the solver. Circuit relies
    upon all objects being in a separate list, which makes the objects input
    particularly handy.

    Periodic boundaries are specified indirectly through V. A handy class for
    defining periodic boundaries is PeriodicBoundary. If all boundaries are
    periodic and there are no objects in the simulation domain the Poisson
    equation has a null space which renders the solution non-unique. To remove
    this null space set remove_null_space=True in the constructor.
    """

    def __init__(self, V, bcs=None, objects=None,
                 circuit=None, remove_null_space=False, eps0=1,
                 linalg_solver='gmres', linalg_precond='hypre_amg'):

        if bcs == None:
            bcs = []

        if objects == None:
            objects = []

        if not isinstance(bcs, list):
            bcs = [bcs]

        if not isinstance(objects, list):
            objects = [objects]

        self.V = V
        self.bcs = bcs
        self.objects = objects
        self.circuit = circuit
        self.remove_null_space = remove_null_space

        """
        One could perhaps identify the cases in which different solvers and preconditioners
        should be used, and by default choose the best suited for the problem.
        """
        self.solver = df.PETScKrylovSolver(linalg_solver, linalg_precond)
        self.solver.parameters['absolute_tolerance'] = 1e-14
        self.solver.parameters['relative_tolerance'] = 1e-12
        self.solver.parameters['maximum_iterations'] = 1000
        self.solver.parameters['nonzero_initial_guess'] = True

        phi = df.TrialFunction(V)
        phi_ = df.TestFunction(V)

        self.a = df.Constant(eps0)*df.inner(df.grad(phi), df.grad(phi_))*df.dx
        A = df.assemble(self.a)

        for bc in bcs:
            bc.apply(A)

        for o in objects:
            o.apply(A)

        if circuit != None:
            A, = circuit.apply(A)

        if remove_null_space:
            phi = df.Function(V)
            null_vec = df.Vector(phi.vector())
            V.dofmap().set(null_vec, 1.0)
            null_vec *= 1.0/null_vec.norm("l2")

            self.null_space = df.VectorSpaceBasis([null_vec])
            df.as_backend_type(A).set_nullspace(self.null_space)

        self.A = A
        self.phi_ = phi_

    def solve(self, rho, bcs=None):

        if bcs == None:
            bcs = []

        if not isinstance(bcs,list):
            bcs = [bcs]

        L = rho*self.phi_*df.dx
        b = df.assemble(L)

        for bc in self.bcs:
            bc.apply(b)

        for o in self.objects:
            o.apply(b)

        if self.circuit != None:
            b, = self.circuit.apply(b)

        for bc in bcs:
            # bc.apply(self.A)    # Could this perchance be removed?
            bc.apply(b)

        # NB: A may not be symmetric in this case. To get it symmetric we need
        # to use assemble_system() but then we need to re-assemble A each time.
        # It was guessed that any speed benefit of having A symmetric would be
        # less than the speed cost of having to re-compute it. Tests welcome.

        if self.remove_null_space:
            self.null_space.orthogonalize(b)

        phi = df.Function(self.V)
        self.solver.solve(self.A, phi.vector(), b)

        return phi

# Faster than the efield-function
class ESolver(object):
    """
    Solves the E-field Equation on the function space V:

        E = - grad phi

    Example:

        solver = PoissonSolver(V, bcs_stationary)
        esolver = ESolver(V)
        phi = solver.solve(rho, bcs)
        E = esolver.solve(phi)

    solve() can be called multiple times while the stiffness matrix is
    assembled only in the constructor to save computations.
    """
    def __init__(self, V, element='CG', degree=1):

        self.V = V

        self.solver = df.PETScKrylovSolver('cg', 'hypre_amg')
        self.solver.parameters['absolute_tolerance'] = 1e-14
        self.solver.parameters['relative_tolerance'] = 1e-12
        self.solver.parameters['maximum_iterations'] = 1000
        self.solver.parameters['nonzero_initial_guess'] = True

        # cell = V.mesh().ufl_cell()
        # W = df.VectorElement("Lagrange", cell, 1)
        W = df.VectorFunctionSpace(V.mesh(), element, degree)
        # V = FiniteElement("Lagrange", cell, 1)
        self.W = W

        E = df.TrialFunction(W)
        E_ = df.TestFunction(W)
        # phi = Coefficient(V)

        self.a = df.inner(E, E_)*df.dx
        self.A = df.assemble(self.a)
        self.E_ = E_

    def solve(self, phi):

        L = df.inner(-df.grad(phi), self.E_)*df.dx
        b = df.assemble(L)

        E = df.Function(self.W)
        self.solver.solve(self.A, E.vector(), b)

        return E

def efield_DG0(mesh, phi):
    Q = df.VectorFunctionSpace(mesh, 'DG', 0)
    q = df.TestFunction(Q)

    M = (1.0 / df.CellVolume(mesh)) * df.inner(-df.grad(phi), q) * df.dx
    E_dg0 = df.assemble(M)
    return df.Function(Q, E_dg0)

class EfieldMean(object):

    def __init__(self, mesh, arithmetic_mean=False):
        assert df.parameters['linear_algebra_backend'] == 'PETSc'
        self.arithmetic_mean = arithmetic_mean
        tdim = mesh.topology().dim()
        gdim = mesh.geometry().dim()
        self.cv = df.CellVolume(mesh)

        self.W = df.VectorFunctionSpace(mesh, 'CG', 1)
        Q = df.VectorFunctionSpace(mesh, 'DG', 0)
        p, self.q = df.TrialFunction(Q), df.TestFunction(Q)
        v = df.TestFunction(self.W)

        ones = df.assemble(
            (1. / self.cv) * df.inner(df.Constant((1,) * gdim), self.q) * df.dx)

        dX = df.dx(metadata={'form_compiler_parameters': {
                   'quadrature_degree': 1, 'quadrature_scheme': 'vertex'}})

        if self.arithmetic_mean:
            A = df.assemble((1./self.cv)*df.Constant(tdim+1)*df.inner(p, v)*dX)
        else:
            A = df.assemble(df.Constant(tdim+1)*df.inner(p, v)*dX)

        Av = df.Function(self.W).vector()
        A.mult(ones, Av)
        Av = df.as_backend_type(Av).vec()
        Av.reciprocal()
        mat = df.as_backend_type(A).mat()
        mat.diagonalScale(L=Av)

        self.A = A

    def mean(self, phi):
        M = (1. / self.cv) * df.inner(-df.grad(phi), self.q) * df.dx
        e_dg0 = df.assemble(M)

        e_field = df.Function(self.W)
        self.A.mult(e_dg0, e_field.vector())
        df.as_backend_type(e_field.vector()).update_ghost_values()
        return e_field

if __name__=='__main__':

    object_type = None
    object_info = []
    n_components = 0   # number of electrical components

    def test_periodic_solver():
        # mesh = Mesh("demos/mesh/rectangle_periodic.xml")
        Lx = 2*df.DOLFIN_PI
        Ly = 2*df.DOLFIN_PI
        Nx = 256
        Ny = 256
        mesh = df.RectangleMesh(df.Point(0,0),df.Point(Lx,Ly),Nx,Ny)

        d = mesh.geometry().dim()
        L = np.empty(2*d)
        for i in range(d):
            l_min = mesh.coordinates()[:,i].min()
            l_max = mesh.coordinates()[:,i].max()
            L[i] = l_min
            L[d+i] = l_max


        PBC = PeriodicBoundary([Lx,Ly])
        V = df.FunctionSpace(mesh, "CG", 1, constrained_domain=PBC)

        class Source(df.Expression):
            def eval(self, values, x):
                values[0] = np.sin(x[0])

        class Exact(df.Expression):
            def eval(self, values, x):
                values[0] = np.sin(x[0])

        f = Source(degree=2)
        phi_e = Exact(degree=2)

        poisson = PoissonSolver(V, remove_null_space=True)
        phi = poisson.solve(f)


        # error_l2 = errornorm(phi_e, phi, "L2")
        # print("l2 norm: ", error_l2)

        vertex_values_phi_e = phi_e.compute_vertex_values(mesh)
        vertex_values_phi = phi.compute_vertex_values(mesh)

        error_max = np.max(vertex_values_phi_e - \
                            vertex_values_phi)
        tol = 1E-9
        msg = 'error_max = %g' %error_max
        print(msg)
        assert error_max < tol , msg

        df.plot(phi, interactive=True)
        df.plot(phi_e, mesh=mesh, interactive=True)


    def test_dirichlet_solver():
        Lx = 1.0
        Ly = 1.0
        Nx = 100
        Ny = 100
        mesh = df.RectangleMesh(df.Point(0,0),df.Point(Lx,Ly),Nx,Ny)
        df.plot(mesh, interactive=True)
        V = df.FunctionSpace(mesh, "CG", 1)
        d = mesh.geometry().dim()

        L = np.empty(2*d)
        for i in range(d):
            l_min = mesh.coordinates()[:,i].min()
            l_max = mesh.coordinates()[:,i].max()
            L[i] = l_min
            L[d+i] = l_max

        u_D = df.Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
        f = df.Constant(-6.0)

        bc = DirichletBC(V, u_D, NonPeriodicBoundary(Ld))

        poisson = PoissonSolver(V, bc)
        phi = poisson.solve(f)

        error_l2 = df.errornorm(u_D, phi, "L2")
        print("l2 norm: ", error_l2)

        vertex_values_u_D = u_D.compute_vertex_values(mesh)
        vertex_values_phi = phi.compute_vertex_values(mesh)

        error_max = np.max(vertex_values_u_D - \
                            vertex_values_phi)
        tol = 1E-10
        msg = 'error_max = %g' %error_max
        assert error_max < tol , msg

        df.plot(phi, interactive=True)

    test_periodic_solver()
    # test_dirichlet_solver()
