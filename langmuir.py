from dolfin import *
from punc import *
import numpy as np
import matplotlib.pyplot as plt
import time

preview = False

show_plot = True if preview else False
store_phi = True

#==============================================================================
# GENERATE MESH
#------------------------------------------------------------------------------

print "Generating mesh"

Lx = 2*DOLFIN_PI
Ly = 2*DOLFIN_PI
Nx = 32
Ny = 32
mesh = RectangleMesh(Point(0,0),Point(Lx,Ly),Nx,Ny)

Nt = 15 if not preview else 1
dt = 0.2

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

multiplicity *= 100

q *= multiplicity
m *= multiplicity

qm = q/m

pop = Population(S)

# Place particls in lattice

#x = np.arange(0,Lx,Lx/Npx)
#y = np.arange(0,Ly,Ly/Npy)
##x = x+0.001*np.sin(x)
#xcart = np.tile(x,Npy)
#ycart = np.repeat(y,Npx)
#pos = np.c_[xcart,ycart]

# Electrons
pos = lp.RandomRectangle(Point(0,0),Point(Lx,Ly)).generate([Npx,Npy])
pos[:,0] += 0.052*np.sin(2*pos[:,0])
qTemp = q[0]*np.ones(len(pos))
qmTemp = qm[0]*np.ones(len(pos))
#pop.add_particles(pos)
pop.add_particles(pos,{'q':qTemp,'qm':qmTemp})

# Ions
pos = lp.RandomRectangle(Point(0,0),Point(Lx,Ly)).generate([Npx,Npy])
qTemp = q[1]*np.ones(len(pos))
qmTemp = qm[1]*np.ones(len(pos))
#pop.add_particles(pos)
pop.add_particles(pos,{'q':qTemp,'qm':qmTemp})

if(False):
	fig = plt.figure()
	pop.scatter(fig)
	fig.suptitle('Initial Particle Position')
	plt.axis([0, Lx, 0, Ly])
	fig.savefig("particles.png")

#==============================================================================
# TIME LOOP
#------------------------------------------------------------------------------

for n in xrange(1,Nt+1):

	print "Computing time-step %d"%n

	#==========================================================================
	# (POS,Q) -> RHO
	#--------------------------------------------------------------------------

	rhoD = Function(D)
	dofmap = D.dofmap()

	## Simply count particles
	#for c in cells(mesh):
	#	cindex = c.index()
	#	dofindex = dofmap.cell_dofs(cindex)[0]
	#	try:
	#		count = len(pop.particle_map[cindex])
	#	except:
	#		count = 0
	##	print cindex, count
	#	rhoD.vector()[dofindex] = count

	for c in cells(mesh):
		cindex = c.index()
		dofindex = dofmap.cell_dofs(cindex)[0]
		cellcharge = 0
		for particle in pop.particle_map[cindex].particles:
			cellcharge += particle.properties['q']
		rhoD.vector()[dofindex] = cellcharge

	rho = project(rhoD,S)

	if(show_plot): plot(rhoD)
	if(show_plot): plot(rho)

	#==========================================================================
	# RHO -> PHI
	#--------------------------------------------------------------------------

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

	# TBD:
	# Possibly replace by fenicstools WeightedGradient.py
	# Need to take check boundary artifacts

	E = project(-grad(phi), V)

	if(show_plot):
		x_hat = Expression(('1.0','0.0'))
		Ex = dot(E,x_hat)
		plot(Ex)

	#==========================================================================
	# E -> (VEL,POS)
	#--------------------------------------------------------------------------

	for particle in pop:
		Ei = E(particle.position)
		fraction = 0.5 if n==1 else 1
		qm = particle.properties['qm']

		particle.velocity += dt*fraction*qm*Ei
		particle.position += dt*particle.velocity
		
		particle.position[0] %= Lx
		particle.position[1] %= Ly

	pop.relocate()

print("Finished")

if(show_plot): interactive()
