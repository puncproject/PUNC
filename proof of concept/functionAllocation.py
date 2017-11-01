"""
Test of how bad it is to use Function(V) to allocate a new variable and return
it vs. to reuse an existing (input) variable
"""

from dolfin import *
import time
import numpy as np

def alternativeA(V, rho, phi):
	phi.vector()[:] = np.array(range(len(phi.vector().array())))

def alternativeB(V, rho):
#	V = rho.function_space()
	phi = Function(V)
	phi.vector()[:] = np.array(range(len(phi.vector().array())))
	return phi


mesh = UnitSquareMesh(64, 64)
V = FunctionSpace(mesh, 'CG', 1)
rho = Function(V)

N = 10000

t = time.time()
phi = Function(V)
for i in xrange(N):
	alternativeA(V, rho, phi)
print(time.time()-t)
print(phi.vector().array()[-10:-1])

t = time.time()
for i in xrange(N):
	phi = alternativeB(rho)
print(time.time()-t)
print(phi.vector().array()[-10:-1])

"""
The conclusion from running this script is that alternative A is faster, which
is as expected. However, alternative B is only a couple of seconds slower for
10000 iterations! The difference is so small that whatever yields the most
beautiful code should be utilized. Computing the function space V inside
alternativeB gives a similar penalty as allocating and returning a new function.
"""
