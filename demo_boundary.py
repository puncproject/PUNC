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

show_plot = True

periodic = np.array([False,True],bool)

nDims = 2						# Number of dimensions
Ns = [4,8,16,32,64]				# Mesh fineness to sweep through
Ld = 2*np.pi*np.ones(nDims)		# Size of domain
Lo = 2*np.ones(nDims)			# Size of object

err = np.zeros(len(Ns))
h = np.zeros(len(Ns))
for i, N in enumerate(Ns):

	Nr = N*np.ones(nDims, dtype=int)
	mesh = RectangleMesh(Point(0,0),Point(Ld),*Nr)
	#plot(mesh)

	constr = PeriodicBoundary(Ld,periodic)
	V = FunctionSpace(mesh, 'CG', 1, constrained_domain=constr)

	# Analytical expressions
	rhoExpr = Expression("2*sin(x[0])*sin(x[1])",degree=1)
	phiExpr = Expression("sin(x[0])*sin(x[1])",degree=1)
	rhoExpr = Expression("sin(x[0])",degree=1)
	phiExpr = Expression("sin(x[0])",degree=1)
	rho = project(rhoExpr,V)

	bnd = NonPeriodicBoundary(Ld,periodic)
	bce = DirichletBC(V,Constant(0),bnd)

	rem = any(periodic==False)
	poisson = PoissonSolver(V,bce,remove_null_space=rem)
	phi = poisson.solve(rho)

	# Compute error
	err[i] = errornorm(phi,project(phiExpr,V),degree_rise=0)
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
