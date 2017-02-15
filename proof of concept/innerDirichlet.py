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

def leftBnd(x, onBnd):
	return near(x[0],-pi) and onBnd

def rightBnd(x, onBnd):
	return near(x[0],pi) and onBnd

def bottomBnd(x, onBnd):
	return near(x[1],-pi) and onBnd

def topBnd(x, onBnd):
	return near(x[1],pi) and onBnd

InteriorBnd = AutoSubDomain(interiorBnd)

leftExpr = Expression("1-pow(x[1]/3.1415,2)",degree=1)
sineExpr = Expression("sin(x[0])*sin(x[1])",degree=1)

bcs = [	DirichletBC(V,0,leftBnd),
		DirichletBC(V,0,rightBnd),
		DirichletBC(V,0,bottomBnd),
		DirichletBC(V,0,topBnd),
		DirichletBC(V,sineExpr,interiorBnd)]

#==============================================================================
# SOLVING BVP ON MAIN MESH
#------------------------------------------------------------------------------

u_ = Function(V)

a=-inner(grad(u),grad(v))*dx
L=-2*sineExpr*v*dx

solve(a==L,u_,bcs=bcs)
plot(u_,interactive=True)

#==============================================================================
# GENERATING SUBMESH
#------------------------------------------------------------------------------

bmesh = BoundaryMesh(mesh, "exterior")
cf = CellFunction("size_t", bmesh, 0)
InteriorBnd = AutoSubDomain(lambda x: np.linalg.norm(x)<2)
InteriorBnd.mark(cf, 1)
submesh = SubMesh(bmesh, cf, 1)

#==============================================================================
# FUNCTION SPACES ON SUBMESH
#------------------------------------------------------------------------------

Vs = FunctionSpace(submesh,'CG',1)
us = TrialFunction(Vs)
vs = TestFunction(Vs)

#==============================================================================
# SOLVING BVP ON SUBMESH (doesn't work)
#------------------------------------------------------------------------------

us_ = Function(Vs)

a = -inner(grad(us),grad(vs))*dx
L = -2*sineExpr*vs*dx

def topPointBnd(x, onBnd):
	return near(x[0],0) and x[1]>0 and onBnd

bcs = []#DirichletBC(Vs,0,topPointBnd)]

solve(a==L,us_,bcs=bcs)
#plot(u_,interactive=True)

coords=submesh.coordinates()
angles=np.arctan2(coords[:,1],coords[:,0])
ind=np.argsort(angles)
ua=us_.vector().array()

#plt.plot(angles[ind],ua[ind])
#plt.show()

print("Full: %f, Bnd: %f"%(u_(0.707,0.707),ua.max()))
