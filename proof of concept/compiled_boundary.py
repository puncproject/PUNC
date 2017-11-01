"""
This demo tests the convergence of the PoissonSolver class with and without
periodic boundary conditions, and with and without objects. A square object has
been used since otherwise, one must take care that the object boundary is
sufficiently refined to not limit the order of the solution. This is a mesh
problem and not a solver problem.
"""

from punc import *
from dolfin import *
from mshr import *
import numpy as np
import copy
import matplotlib.pyplot as plt
import time

n_dims = 2						# Number of dimensions
Ld = 2*np.pi*np.ones(n_dims)	# Size of domain

Nr = 256*np.ones(n_dims,int)
mesh = RectangleMesh(Point(0,0),Point(*Ld),*Nr)

V = FunctionSpace(mesh, 'CG', 1)

# Analytical expressions
rho_expr = Expression("2*sin(x[0])*sin(x[1])",degree=1)
phi_expr = Expression("sin(x[0])*sin(x[1])",degree=1)
rho = project(rho_expr,V)

bce = DirichletBC(V,phi_expr,NonPeriodicBoundary(Ld))

bc_str = "(near(x[0],0) || near(x[0],Ldx) || near(x[1],0) || near(x[1],Ldy)) && on_boundary"
bc_sub = CompiledSubDomain(bc_str,Ldx=Ld[0],Ldy=Ld[1])
bce_c = DirichletBC(V,phi_expr,bc_sub)

poisson = PoissonSolver(V)

L = rho*poisson.phi_*dx
b = assemble(L)

N = 1000

t = time.time()
for i in range(N):
	# phi_c = poisson.solve(rho,bce_c)
	bce_c.apply(poisson.A)
	bce_c.apply(b)
print("Compiled subdomain:   %f s"%(time.time()-t))

t = time.time()
for i in range(N):
	# phi = poisson.solve(rho,bce)
	bce.apply(poisson.A)
	bce.apply(b)
print("Uncompiled subdomain: %f s"%(time.time()-t))

t = time.time()
for i in range(N):
	# phi_c = poisson.solve(rho,bce_c)
	bce_c.apply(poisson.A)
	bce_c.apply(b)
print("Compiled subdomain:   %f s"%(time.time()-t))
