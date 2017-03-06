# __authors__ = ('Sigvald Marholm <sigvaldm@fys.uio.no>')
# __date__ = '2017-02-22'
# __copyright__ = 'Copyright (C) 2017' + __authors__
# __license__  = 'GNU Lesser GPL version 3 or any later version'

from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
	from itertools import izip as zip
	range = xrange

#import dolfin as df
from dolfin import *
import numpy as np

class PeriodicBoundary(SubDomain):

	def __init__(self, Ld):
		SubDomain.__init__(self)
		self.Ld = Ld

	# Target domain
	def inside(self, x, onBnd):
		return bool(		any([near(a,0) for a in x])					# On any lower bound
					and not any([near(a,b) for a,b in zip(x,self.Ld)])	# But not any upper bound
					and onBnd)

	# Map upper edges to lower edges
	def map(self, x, y):
		y[:] = [a-b if near(a,b) else a for a,b in zip(x,self.Ld)]

class PoissonSolver:

	def __init__(self, V):

		self.solver = PETScKrylovSolver('gmres', 'hypre_amg')
		self.solver.parameters['absolute_tolerance'] = 1e-14
		self.solver.parameters['relative_tolerance'] = 1e-12
		self.solver.parameters['maximum_iterations'] = 1000

		self.V = V

		phi = TrialFunction(V)
		phi_ = TestFunction(V)

		a = inner(nabla_grad(phi), nabla_grad(phi_))*dx
		A = assemble(a)

		self.solver.set_operator(A)
		self.phi_ = phi_

		phi = Function(V)
		null_vec = Vector(phi.vector())
		V.dofmap().set(null_vec, 1.0)
		null_vec *= 1.0/null_vec.norm("l2")

		self.null_space = VectorSpaceBasis([null_vec])
		as_backend_type(A).set_nullspace(self.null_space)

	def solve(self, rho):

		L = rho*self.phi_*dx
		b = assemble(L)
		self.null_space.orthogonalize(b);

		phi = Function(self.V)
		self.solver.solve(phi.vector(), b)

		return phi

def EField(phi):
	V = phi.ufl_function_space()
	mesh = V.mesh()
	degree = V.ufl_element().degree()
	constr = V.constrained_domain
	W = VectorFunctionSpace(mesh, 'CG', degree, constrained_domain=constr)
	return project(-grad(phi), W)
