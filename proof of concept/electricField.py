from dolfin import *
import numpy as np
from numpy import pi
from matplotlib import pyplot as plt

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

nDims = 2							# Number of dimensions
Ld = 2*pi*np.ones(nDims)			# Length of domain

err1 = []
err2 = []
Ns = [4,8,16,32,64]
for N in Ns:

	print(N)

	Nr = N*np.ones(nDims,dtype=int)	# Number of 'rectangles' in mesh

	mesh = RectangleMesh(Point(0,0),Point(Ld),*Nr)
	V = FunctionSpace(mesh, 'CG', 1, constrained_domain=PeriodicBoundary(Ld))
	W1 = VectorFunctionSpace(mesh, 'CG', 1)
	W2 = VectorFunctionSpace(mesh, 'CG', 1, constrained_domain=PeriodicBoundary(Ld))

	theta = 1.0 # theta=0 gives unrealistically favourable conditions
	phi = project(Expression("sin(x[0]+theta)",degree=1,theta=theta),V)
	E1 = project(-grad(phi), W1)
	E2 = project(-grad(phi), W2)
	Ee1 = project(Expression(("-cos(x[0]+theta)","0"),degree=1,theta=theta),W1)
	Ee2 = project(Expression(("-cos(x[0]+theta)","0"),degree=1,theta=theta),W2)

	err1.append(np.max(E1.vector().array()-Ee1.vector().array()))
	err2.append(np.max(E2.vector().array()-Ee2.vector().array()))

delta = Ld[0]/Ns

plt.loglog(delta,err1,label="Unconstrained")
plt.loglog(delta,err2,label="Constrained")
plt.legend(loc="upper left")
plt.xlabel("Spatial stepsize")
plt.ylabel("Max. norm")
plt.grid()
plt.show()

order1 = np.log(err1[-1]/err1[-2])/np.log(delta[-1]/delta[-2])
order2 = np.log(err2[-1]/err2[-2])/np.log(delta[-1]/delta[-2])

print("order1: %.2f"%order1)
print("order2: %.2f"%order2)
