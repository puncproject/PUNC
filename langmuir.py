from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import time
from punc import *
from WeightedGradient import weighted_gradient_matrix

preview = False
cgspace = True
lattice = False

show_plot = True if preview else False
store_phi = False
set_log_level(WARNING)

np.random.seed(666)

#==============================================================================
# GENERATE MESH
#------------------------------------------------------------------------------

print "Generating mesh"

Lx = 2*DOLFIN_PI
Ly = 2*DOLFIN_PI
Lxy = np.array([Lx,Ly])
Nx = 32
Ny = Nx
deltax = Lx/(Nx-1)	# assumes periodic
deltay = Ly/(Ny-1)	# assumes periodic
da = deltax*deltay if cgspace else deltax*deltay*0.5
mesh = RectangleMesh(Point(0,0),Point(Lx,Ly),Nx,Ny)

Nt = 25 if not preview else 1
dt = 0.0251327

if(show_plot): plot(mesh)

#==============================================================================
# PLOTTING FUNCTION
#------------------------------------------------------------------------------

def myPlot(u,fName):
	mesh = u.function_space().mesh()
	x = mesh.coordinates()[:,0]
	v = u.compute_vertex_values(mesh)
	y = mesh.coordinates()[:,1]
	t = mesh.cells()

	fig = plt.figure(figsize=(5,5))
	ax  = fig.add_subplot(111)

	c = ax.tricontourf(x, y, t, v)
	fig.savefig(fName)

#==============================================================================
# FUNCTION SPACE
#------------------------------------------------------------------------------

print "Generating function spaces"

class PeriodicBoundary(SubDomain):

	# Target domain
	def inside(self, x, on_bnd):
		return bool(		(near(x[0],0)  or near(x[1],0))	 # On a lower bound
					and not (near(x[0],Lx) or near(x[1],Ly)) # But not an upper
					and on_bnd)

	# Map upper edges to lower edges
	def map(self, x, y):
		y[0] = x[0]-Lx if near(x[0],Lx) else x[0]
		y[1] = x[1]-Ly if near(x[1],Ly) else x[1]

constrained_domain=PeriodicBoundary()

D =       FunctionSpace(mesh, 'DG', 0, constrained_domain=constrained_domain)
S =       FunctionSpace(mesh, 'CG', 1, constrained_domain=constrained_domain)
V = VectorFunctionSpace(mesh, 'CG', 1, constrained_domain=constrained_domain)

#==============================================================================
# INITIALIZE SOLVER
#------------------------------------------------------------------------------

print "Initializing field solver"

if not has_linear_algebra_backend("PETSc"):
    info("langmuir.py needs PETSc")
    exit()

parameters["linear_algebra_backend"] = "PETSc"

phi = TrialFunction(S)
phi_ = TestFunction(S)

a = inner(nabla_grad(phi), nabla_grad(phi_))*dx
A = assemble(a)

solver = PETScKrylovSolver("cg")
solver.set_operator(A)

phi = Function(S)

null_vec = Vector(phi.vector())
S.dofmap().set(null_vec, 1.0)
null_vec *= 1.0/null_vec.norm("l2")

null_space = VectorSpaceBasis([null_vec])
as_backend_type(A).set_nullspace(null_space)

#==============================================================================
# INITIALIZE PARTICLES
#------------------------------------------------------------------------------

print "Initializing particles"

Npmul = 4 if preview else 1
Npx = Nx*4*Npmul
Npy = Ny*4*Npmul
Np = Npx*Npy

q = np.array([-1., 1.])
m = np.array([1., 1836.])

multiplicity = (Lx*Ly/Np)*m[0]/(q[0]**2)

#multiplicity *= 100

q *= multiplicity
m *= multiplicity

qm = q/m

pop = Population(S,V)

# Place particls in lattice

if(lattice):
	x = np.arange(0,Lx,Lx/Npx)
	y = np.arange(0,Ly,Ly/Npy)
	x += (0.001/deltax)*np.sin(x)
	xcart = np.tile(x,Npy)
	ycart = np.repeat(y,Npx)
	pos = np.c_[xcart,ycart]
	qTemp = q[0]*np.ones(len(pos))
	mTemp = m[0]*np.ones(len(pos))
	qmTemp = qm[0]*np.ones(len(pos))
	pop.addParticles(pos,{'q':qTemp,'qm':qmTemp,'m':mTemp})

	x = np.arange(0,Lx,Lx/Npx)
	y = np.arange(0,Ly,Ly/Npy)
	xcart = np.tile(x,Npy)
	ycart = np.repeat(y,Npx)
	pos = np.c_[xcart,ycart]
	qTemp = q[1]*np.ones(len(pos))
	mTemp = m[1]*np.ones(len(pos))
	qmTemp = qm[1]*np.ones(len(pos))
	pop.addParticles(pos,{'q':qTemp,'qm':qmTemp,'m':mTemp})

else:

	# Electrons
	pos = RandomRectangle(Point(0,0),Point(Lx,Ly)).generate([Npx,Npy])
	pos[:,0] += (0.001/deltax)*np.sin(pos[:,0])
	qTemp = q[0]*np.ones(len(pos))
	mTemp = m[0]*np.ones(len(pos))
	qmTemp = qm[0]*np.ones(len(pos))
	pop.addParticles(pos,{'q':qTemp,'qm':qmTemp,'m':mTemp})

	# Ions
	pos = RandomRectangle(Point(0,0),Point(Lx,Ly)).generate([Npx,Npy])
	qTemp = q[1]*np.ones(len(pos))
	mTemp = m[1]*np.ones(len(pos))
	qmTemp = qm[1]*np.ones(len(pos))
	pop.addParticles(pos,{'q':qTemp,'qm':qmTemp,'m':mTemp})

if(False):
	fig = plt.figure()
	pop.scatter(fig)
	fig.suptitle('Initial Particle pos')
	plt.axis([0, Lx, 0, Ly])
	fig.savefig("particles.png")

tPush = 0

#==============================================================================
# TIME LOOP
#------------------------------------------------------------------------------

KE = np.zeros(Nt)
PE = np.zeros(Nt)

for n in xrange(1,Nt+1):

	print "Computing time-step %d"%n

	#==========================================================================
	# (POS,Q) -> RHO
	#--------------------------------------------------------------------------

	print "    Accumulating charges"

	rho = Function(S)
	if(not cgspace):
		rho = distrDG0(pop,rho,D,S)
	else:
		distrCG1(pop,rho,da)

	if(show_plot): plot(rhoD)
	if(show_plot): plot(rho)

	#==========================================================================
	# RHO -> PHI
	#--------------------------------------------------------------------------

	print "    Solving potential"

	L = rho*phi_*dx
	b = assemble(L)
	null_space.orthogonalize(b);

	solver.solve(phi.vector(), b)

	if(show_plot): plot(phi)

	if(store_phi):
		myPlot(phi,"png/phi_%d"%n)
#		viz = plot(phi,key="u")
#		viz.write_png("png/phi_%d"%n)
#		file = File("vtk/phi_%d.pvd"%n)
#		file << phi

	#==========================================================================
	# PHI -> E
	#--------------------------------------------------------------------------

	print "    Gradient of potential"

	# TBD:
	# Possibly replace by fenicstools WeightedGradient.py
	# Need to take check boundary artifacts

	E = project(-grad(phi), V)
	'''
	dP = weighted_gradient_matrix(mesh, (0,1), 1, constrained_domain=constrained_domain)
	Ex = Function(S)
	Ey = Function(S)
	Ex.vector()[:] = dP[0] * phi.vector()
	Ey.vector()[:] = dP[1] * phi.vector()
	'''
	if(show_plot):
		x_hat = Expression(('1.0','0.0'))
		Ex = dot(E,x_hat)
		plot(Ex)

	#==========================================================================
	# E -> (VEL,POS)
	#--------------------------------------------------------------------------

	print "    Pushing particles"

	fraction = 0.5 if n==1 else 1.0
	KE[n-1] = accel(pop,E,dt*fraction)
	PE[n-1] = potEnergy(pop,phi)
	movePeriodic(pop,dt,Lxy)

	pop.relocate()

KE[0]=0

fig = plt.figure()
plt.plot(range(Nt),KE)
plt.plot(range(Nt),PE)
plt.plot(range(Nt),PE+KE)
plt.savefig('energy.png')

TE = KE+PE
TEdev = TE - TE[0]
exdev = np.max(np.abs(TEdev))/TE[0]

print "max relative deviation: ", 100*exdev, "%"


print("Finished")

if(show_plot): interactive()
