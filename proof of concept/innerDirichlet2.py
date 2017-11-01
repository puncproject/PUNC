from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pylab as plt

pi = DOLFIN_PI

#==============================================================================
# GENERATE MESH
#------------------------------------------------------------------------------

R = 1
domain = Rectangle(Point(-pi,-pi),Point(pi,pi))-Circle(Point(0,0), R)
mesh = generate_mesh(domain, 32)

#==============================================================================
# FUNCTION SPACES
#------------------------------------------------------------------------------

V = FunctionSpace(mesh,'CG',1)
u = TrialFunction(V)
v = TestFunction(V)

#==============================================================================
# BOUNDARY CONDITIONS
#------------------------------------------------------------------------------

def interiorBnd(x, onBnd):
	return np.linalg.norm(x)<R+DOLFIN_EPS and onBnd

def exteriorBnd(x, onBnd):
	return any(map(lambda y: near(y,pi),np.abs(x))) and onBnd

def anyBnd(x, onBnd):
	return interiorBnd(x, onBnd) or exteriorBnd(x,onBnd)

InteriorBnd = AutoSubDomain(interiorBnd)

sineExpr = Expression("sin(x[0])*sin(x[1])",degree=1)

bcs = DirichletBC(V,sineExpr,anyBnd)

#==============================================================================
# SOLVING BVP ON MAIN MESH
#------------------------------------------------------------------------------

u_ = Function(V)
u_.vector().array()[0] = 10

a=-inner(grad(u),grad(v))*dx
L=-2*sineExpr*v*dx

# SOLUTION METHOD 1 

#solve(a==L,u_,bcs=bcs)
#plot(u_,interactive=True)

# BC METHOD 1 FOR SOLUTION METHODS 2 AND 3
A = assemble(a)
bcs.apply(A)

b = assemble(L)
bcs.apply(b)

# BC METHOD 2 (equivalent to 1)
#A, b = assemble_system(a, L, bcs)

# SOLUTION METHOD 2 (can't choose solver)

#solve(A,u_.vector(),b)

# SOLUTION METHOD 3

# Choose either solver (that converges)
#solver = PETScLUSolver() # Converges
solver = PETScKrylovSolver("gmres") # Converges
#solver = PETScKrylovSolver("cg") # Doesn't converge


solver.set_operator(A)

solver.solve(u_.vector(), b)
plot(u_,interactive=True)

