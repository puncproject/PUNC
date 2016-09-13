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

Ld = 2*DOLFIN_PI*np.array([1,1])	# Length of domain
Nc = 32*np.array([1,1])				# Number of cells
delta = Ld/Nc						# Stepsizes
Nt = 25 if not preview else 1		# Number of time steps
dt = 0.0251327						# Time step
Np = Nc*(16 if preview else 4)		# Number of particles 

mesh = RectangleMesh(Point(0,0),Point(Ld),Nc[0],Nc[1])

if(show_plot): plot(mesh)

#==============================================================================
# FUNCTION SPACE
#------------------------------------------------------------------------------

print "Generating function spaces"

constrained_domain=PeriodicBoundary(Ld)

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

pop = Population(S,V)

if(lattice):
	pop.addLatticeSine(Np,Ld,0.001/delta[0])
else:
	pop.addRandomSine(Np,Ld,0.001/delta[0])


if(False):
	fig = plt.figure()
	pop.scatter(fig)
	fig.suptitle('Initial Particle pos')
	plt.axis([0, Lx, 0, Ly])
	fig.savefig("particles.png")

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
		distrCG1(pop,rho,np.prod(delta))

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
	movePeriodic(pop,dt,Ld)

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
