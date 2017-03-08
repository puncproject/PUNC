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

class PeriodicBoundary(df.SubDomain):

    def __init__(self, Ld):
        df.SubDomain.__init__(self)
        self.Ld = Ld

    # Target domain
    def inside(self, x, onBnd):
        return bool(        any([df.near(a,0) for a in x])                  # On any lower bound
                    and not any([df.near(a,b) for a,b in zip(x,self.Ld)])   # But not any upper bound
                    and onBnd)

    # Map upper edges to lower edges
    def map(self, x, y):
        y[:] = [a-b if df.near(a,b) else a for a,b in zip(x,self.Ld)]

def dirichlet_bcs(V, facet_f, n_components = 0, phi0 = df.Constant(0.0), E0 = None):

    d = V.mesh().geometry().dim()
    if E0 is not None:
        if d == 2:
            phi0 = 'x[0]*Ex + x[1]*Ey'
            phi0 = df.Expression(phi0, degree = 1, Ex = -E0[0], Ey = -E0[1])
        if d == 3:
            phi0 = 'x[0]*Ex + x[1]*Ey + x[2]*Ez'
            phi0 = df.Expression(phi0, degree = 1,
                              Ex = -E0[0], Ey = -E0[1], Ez = -E0[2])
    bcs = []
    for i in range(2*d):
        bc0 = df.DirichletBC(V, phi0, facet_f, (n_components+i+1))
        bcs.append(bc0)
    return bcs

class PoissonSolverPeriodic:

    def __init__(self, V):

        self.solver = df.PETScKrylovSolver('gmres', 'hypre_amg')
        self.solver.parameters['absolute_tolerance'] = 1e-14
        self.solver.parameters['relative_tolerance'] = 1e-12
        self.solver.parameters['maximum_iterations'] = 1000

        self.V = V

        phi = df.TrialFunction(V)
        phi_ = df.TestFunction(V)

        a = df.inner(df.nabla_grad(phi), df.nabla_grad(phi_))*df.dx
        A = df.assemble(a)

        self.solver.set_operator(A)
        self.phi_ = phi_

        phi = df.Function(V)
        null_vec = df.Vector(phi.vector())
        V.dofmap().set(null_vec, 1.0)
        null_vec *= 1.0/null_vec.norm("l2")

        self.null_space = df.VectorSpaceBasis([null_vec])
        df.as_backend_type(A).set_nullspace(self.null_space)

    def solve(self, rho, object_bcs = None):

        L = rho*self.phi_*df.dx

        if object_bcs is None:
            b = df.assemble(L)
        else:
            A, b = df.assemble_system(self.a, L, object_bcs)

        self.null_space.orthogonalize(b)

        phi = df.Function(self.V)
        self.solver.solve(phi.vector(), b)

        return phi

class PoissonSolverDirichlet:

    def __init__(self, V, bcs):

        self.solver = df.PETScKrylovSolver('gmres', 'hypre_amg')
        self.solver.parameters['absolute_tolerance'] = 1e-14
        self.solver.parameters['relative_tolerance'] = 1e-12
        self.solver.parameters['maximum_iterations'] = 1000

        self.V = V
        self.bcs = bcs

        phi = df.TrialFunction(V)
        phi_ = df.TestFunction(V)

        a = df.inner(df.nabla_grad(phi), df.nabla_grad(phi_))*df.dx
        self.A = df.assemble(a)
        [bc.apply(self.A) for bc in self.bcs]

        self.phi_ = phi_

    def solve(self, rho, object_bcs = None):

        L = rho*self.phi_*df.dx
        b = df.assemble(L)
        [bc.apply(b) for bc in self.bcs]

        phi = df.Function(self.V)
        if object_bcs is None:
            self.solver.solve(self.A, phi.vector(), b)
        else:
            [bc.apply(self.A) for bc in object_bcs]
            [bc.apply(b) for bc in object_bcs]

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
    return df.project(-df.grad(phi), W)

if __name__=='__main__':

    from mark_object import *

    object_type = None
    object_info = []
    n_components = 0   # number of electrical components

    def test_periodic_solver():
        # mesh = Mesh("demos/mesh/rectangle_periodic.xml")
        Lx = 2*DOLFIN_PI
        Ly = 2*DOLFIN_PI
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
                values[0] = sin(x[0])

        class Exact(df.Expression):
            def eval(self, values, x):
                values[0] = sin(x[0])

        f = Source(degree=2)
        phi_e = Exact(degree=2)

        poisson = PoissonSolverPeriodic(V)
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

        facet_f = mark_boundaries(mesh, L, object_type, object_info, n_components)
        plot(facet_f, interactive=True)

        bcs = dirichlet_bcs(V, facet_f, n_components, phi0 = u_D)

        poisson = PoissonSolverDirichlet(V, bcs)
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
