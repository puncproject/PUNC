from dolfin import *

#==============================================================================
# MESH
#------------------------------------------------------------------------------

Lx = 2*DOLFIN_PI
Ly = 2*DOLFIN_PI
Nx = 16
Ny = 16
mesh = RectangleMesh(Point(0,0),Point(Lx,Ly),Nx,Ny)

plot(mesh)

#==============================================================================
# FUNCTION SPACE
#------------------------------------------------------------------------------

class PeriodicBoundary(SubDomain):

	# Target domain
	def inside(self, x, on_bnd):
		return bool(		(near(x[0],0)  or near(x[1],0))	 # Near a lower bnd
					and not (near(x[0],Lx) or near(x[1],Ly)) # But not touching an upper bnd
					and on_bnd)

	# Map upper edges to lower edges
	def map(self, x, y):
		y[0] = x[0]-Lx if near(x[0],Lx) else x[0]
		y[1] = x[1]-Ly if near(x[1],Ly) else x[1]

constrained_domain=PeriodicBoundary()

S = FunctionSpace(mesh, 'CG', 1, constrained_domain=constrained_domain)
V = VectorFunctionSpace(mesh, 'CG', 1)

#==============================================================================
# RHO -> PHI
#------------------------------------------------------------------------------

rho = Expression('sin((2*DOLFIN_PI/%f)*x[0])'%Lx)
plot(project(rho,S))

#==============================================================================
# RHO -> PHI
#------------------------------------------------------------------------------

# Fix undetermined constant in phi
bc = DirichletBC(S,Constant(0),"near(x[0],0)")
#bc = []

phi = TrialFunction(S)
phi_ = TestFunction(S)

a = inner(nabla_grad(phi), nabla_grad(phi_))*dx
L = rho*phi_*dx

phi = Function(S)
solve(a == L, phi, bc)

plot(phi)

#==============================================================================
# PHI -> E
#------------------------------------------------------------------------------

# TBD:
# Possibly replace by fenicstools WeightedGradient.py
# Need to take care of boundary artifacts

E = project(-grad(phi), V)

x_hat = Expression(('1.0','0.0'))
Ex = dot(E,x_hat)

plot(Ex)

interactive()
