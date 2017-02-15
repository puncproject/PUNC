from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pylab as plt

pi = DOLFIN_PI

#==============================================================================
# GENERATE MESH
#------------------------------------------------------------------------------

R = 1
domain = Rectangle(Point(-pi,-pi),Point(pi,pi))-Circle(Point(0,0), R, 8)
mesh = generate_mesh(domain, 4)
#plot(mesh)

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
# GENERATING MAPPING INDICES
#------------------------------------------------------------------------------

#bmesh = BoundaryMesh(mesh, "exterior")
#cf = CellFunction("size_t", bmesh, 0)
cf = VertexFunction("size_t", mesh, 0)
InteriorBnd = AutoSubDomain(lambda x: np.linalg.norm(x)<2)
InteriorBnd.mark(cf, 1)
coords = mesh.coordinates()
plot(cf)
#submesh = SubMesh(bmesh, cf, 1)

#==============================================================================
# SOLVING BVP ON MAIN MESH
#------------------------------------------------------------------------------

u_ = Function(V)

a=-inner(grad(u),grad(v))*dx
L=-2*sineExpr*v*dx

A = assemble(a)
bcs.apply(A)

b = assemble(L)
bcs.apply(b)

solver = PETScKrylovSolver("gmres")
solver.set_operator(A)
solver.solve(u_.vector(), b)

#plot(u_,interactive=True)

