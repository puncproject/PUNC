# __authors__ = ('Sigvald Marholm <sigvaldm@fys.uio.no>')
# __date__ = '2017-02-22'
# __copyright__ = 'Copyright (C) 2017' + __authors__
# __license__  = 'GNU Lesser GPL version 3 or any later version'

from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
    from itertools import izip as zip
    range = xrange

import dolfin as df
import numpy as np

def load_mesh(fname):
    mesh   = df.Mesh(fname+".xml")
    boundaries = df.MeshFunction("size_t", mesh, fname+"_facet_region.xml")
    return mesh, boundaries

def unit_mesh(N):
	"""
	Given a list of cell divisions, N, returns a mesh with unit size in each
	spatial direction.
	"""
	d = len(N)
	mesh_types = [df.UnitIntervalMesh,
	     		  df.UnitSquareMesh,
				  df.UnitCubeMesh]

	return mesh_types[d-1](*N)

def simple_mesh(Ld, N):
	"""
	Returns a mesh for a given list, Ld, containing the size of domain in each
	spatial direction, and the corresponding number of cell divisions N.
	"""
	d = len(N)
	mesh_types = [df.RectangleMesh, df.BoxMesh]

	return mesh_types[d-2](df.Point(0,0,0), df.Point(*Ld), *N)

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
    may work. bcs can be a single DirichletBC object or Object object or a list
    of such.

    Periodic boundaries are specified indirectly through V. A handy class for
    defining periodic boundaries is PeriodicBoundary. If all boundaries are
    periodic and there are no objects in the simulation domain the Poisson
    equation has a null space which renders the solution non-unique. To remove
    this null space set remove_null_space=True in the constructor.
    """

    def __init__(self, V, bcs=[], remove_null_space=False):

        # Make sure bcs is an iterable list
        if isinstance(bcs,df.fem.bcs.DirichletBC):
            bcs = [bcs]

        if bcs == None:
            bcs = []

        self.V = V
        self.bcs = bcs
        self.remove_null_space = remove_null_space

        self.solver = df.PETScKrylovSolver('gmres', 'hypre_amg')
        self.solver.parameters['absolute_tolerance'] = 1e-14
        self.solver.parameters['relative_tolerance'] = 1e-12
        self.solver.parameters['maximum_iterations'] = 1000
        self.solver.parameters['nonzero_initial_guess'] = True

        phi = df.TrialFunction(V)
        phi_ = df.TestFunction(V)

        self.a = df.inner(df.nabla_grad(phi), df.nabla_grad(phi_))*df.dx
        A = df.assemble(self.a)

        for bc in bcs:
            bc.apply(A)

        if remove_null_space:
            phi = df.Function(V)
            null_vec = df.Vector(phi.vector())
            V.dofmap().set(null_vec, 1.0)
            null_vec *= 1.0/null_vec.norm("l2")

            self.null_space = df.VectorSpaceBasis([null_vec])
            df.as_backend_type(A).set_nullspace(self.null_space)

        self.A = A
        self.phi_ = phi_

    def solve(self, rho, bcs=[]):

        # Make sure bcs is an iterable list
        if isinstance(bcs,df.fem.bcs.DirichletBC):
            bcs = [bcs]

        if bcs == None:
            bcs = []

        L = rho*self.phi_*df.dx
        b = df.assemble(L)

        for bc in self.bcs:
            bc.apply(b)

        for bc in bcs:
            bc.apply(self.A)
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

def electric_field(phi):
    """
    This function calculates the gradient of the electric potential, which
    is the electric field:

            E = -\del\varPhi

    Args:
          phi   : The electric potential.

    returns:
          E: The electric field.
    """
    V = phi.ufl_function_space()
    mesh = V.mesh()
    degree = V.ufl_element().degree()
    constr = V.constrained_domain
    W = df.VectorFunctionSpace(mesh, 'CG', degree, constrained_domain=constr)
    return df.project(-df.grad(phi), W, solver_type="gmres",
                       preconditioner_type="petsc_amg")

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
