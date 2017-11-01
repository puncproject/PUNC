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

show_plot = False

"""
The solver should work for all of the following cases. Note that the null space
should only be removed when no boundaries are dirichlet, that is, periodic
boundary in all directions and no objects.
"""
# periodic, with_object, remove_null_space = True, False, True
periodic, with_object, remove_null_space = True, True, False
# periodic, with_object, remove_null_space = False, False, False
# periodic, with_object, remove_null_space = False, True, False

n_dims = 2						# Number of dimensions
Ns = [4,8,16,32,64,128]			# Mesh fineness to sweep through
Ld = 2*np.pi*np.ones(n_dims)		# Size of domain
Lo = 2*np.ones(n_dims)			# Size of object

def exteriorBnd(x, on_bnd):
	return on_bnd and (
		np.any([near(a,0) for a in x]) or
		np.any([near(a,b) for a,b in zip(x,Ld)]))

def interiorBnd(x, on_bnd):
	return on_bnd and (
	np.any([near(a,b) for a,b in zip(x,(Ld+Lo)/2)]) or
	np.any([near(a,b) for a,b in zip(x,(Ld-Lo)/2)]))

err = np.zeros(len(Ns))
h = np.zeros(len(Ns))
for i, N in enumerate(Ns):

	if with_object:
		domain = Rectangle(Point(0,0),Point(Ld)) - Rectangle(Point((Ld-Lo)/2),Point((Ld+Lo)/2))
		mesh = generate_mesh(domain,N)
	else:
		Nr = N*np.ones(n_dims, dtype=int)
		mesh = RectangleMesh(Point(0,0),Point(Ld),*Nr)
	#plot(mesh)

	constr = PeriodicBoundary(Ld) if periodic else None
	V = FunctionSpace(mesh, 'CG', 1, constrained_domain=constr)

	# Analytical expressions
	rho_expr = Expression("2*sin(x[0])*sin(x[1])",degree=1)
	phi_expr = Expression("sin(x[0])*sin(x[1])",degree=1)
	rho = project(rho_expr,V)

	bce = DirichletBC(V,phi_expr,exteriorBnd) if not periodic else None
	poisson = PoissonSolver(V,bce,remove_null_space=remove_null_space)

	# The Poisson equation is solved twice with different boundary conditions.
	# This is to ensure the solver doesn't break when using dynamically varying
	# boundary conditions, e.g. that any matrix or vector is broken when bcs
	# are applied several times.

	bci = [DirichletBC(V,2*phi_expr,interiorBnd)] if with_object else None
	phi = poisson.solve(2*rho,bci)

	bci = [DirichletBC(V,phi_expr,interiorBnd)] if with_object else []
	phi = poisson.solve(rho,bci)

	# Compute error
	err[i] = errornorm(phi,project(phi_expr,V),degree_rise=0)
	h[i] = mesh.hmin()
	order = ln(err[i]/err[i-1])/ln(h[i]/h[i-1]) if i>0 else 0
	print("Running with N=%3d: h=%2.2E, E=%2.2E, order=%2.2E"%(N,h[i],err[i],order))

	if i==len(Ns)-1 and show_plot:
		#plot(rho)
		plot(phi)
		interactive()

if show_plot:
	plt.loglog(h,err)
	plt.grid()
	plt.title('Convergence of PUNC PoissonSolver class')
	plt.xlabel('Minimum cell diameter in mesh')
	plt.ylabel('L2 error norm')
	plt.show()
