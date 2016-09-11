from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import time
from punc import *
from WeightedGradient import weighted_gradient_matrix

preview = False
cgspace = False
lattice = False

show_plot = True if preview else False
store_phi = True
set_log_level(WARNING)

np.random.seed(666)

#==============================================================================
# GENERATE MESH
#------------------------------------------------------------------------------

print "Generating mesh"

Lx = 2*DOLFIN_PI
Ly = 2*DOLFIN_PI
Nx = 32
Ny = Nx
deltax = Lx/(Nx-1)	# assumes periodic
deltay = Ly/(Ny-1)	# assumes periodic
mesh = RectangleMesh(Point(0,0),Point(Lx,Ly),Nx,Ny)

Nt = 50 if not preview else 1
dt = 0.251

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

KE = np.zeros(Nt+1)
PE = np.zeros(Nt+1)

for n in xrange(1,Nt+1):

	print "Computing time-step %d"%n

	#==========================================================================
	# (POS,Q) -> RHO
	#--------------------------------------------------------------------------

	print "    Accumulating charges"

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

	## Add up different charges, dictionary variant
	#for c in cells(mesh):
	#	cindex = c.index()
	#	dofindex = dofmap.cell_dofs(cindex)[0]
	#	cellcharge = 0
	#	for particle in pop.particle_map[cindex].particles:
	#		cellcharge += particle.properties['q']
	#	rhoD.vector()[dofindex] = cellcharge

	if(not cgspace):
		# Add up different charges, list variant
		for c in cells(mesh):
			cindex = c.index()
			dofindex = dofmap.cell_dofs(cindex)[0]
			cellcharge = 0
			for particle in pop[cindex]:
				cellcharge += particle.properties['q']/(deltax*deltay)
			rhoD.vector()[dofindex] = cellcharge

		rho = project(rhoD,S)
	else:
		dofmap = S.dofmap()
		rho = Function(S) # zero

		for c in cells(mesh):
			cindex = c.index()
			dofindex = dofmap.cell_dofs(cindex)

			accum = np.zeros(3)
			for p in pop[cindex]:

				pop.Selement.evaluate_basis_all(	pop.Sbasis_matrix,
													p.pos,
													c.get_vertex_coordinates(),
													c.orientation())

				q=p.properties['q']
				accum += (q/(deltax*deltay))*pop.Sbasis_matrix.T[0]

			rho.vector()[dofindex] += accum

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

	t = df.Timer("push")


	for c in cells(mesh):
		cindex = c.index()
		E.restrict(		pop.Vcoefficients,
						pop.Velement,
						c,
						c.get_vertex_coordinates(),
						c)
		phi.restrict(	pop.Scoefficients,
						pop.Selement,
						c,
						c.get_vertex_coordinates(),
						c)

		for p in pop[cindex]:
			pop.Velement.evaluate_basis_all(	pop.Vbasis_matrix,
												p.pos,
												c.get_vertex_coordinates(),
												c.orientation())
			pop.Selement.evaluate_basis_all(	pop.Sbasis_matrix,
												p.pos,
												c.get_vertex_coordinates(),
												c.orientation())

			Ei = np.dot(pop.Vcoefficients, pop.Vbasis_matrix)[:]
			phii = np.dot(pop.Scoefficients, pop.Sbasis_matrix)[:]

			q = p.properties['q']
			m = p.properties['m']
			qm = p.properties['qm']

			fraction = 0.5 if n==1 else 1
			inc = dt*fraction*qm*Ei
			vel = p.vel

			KE[n] += 0.5*m*np.dot(vel,vel+inc)

			p.vel += inc
			p.pos += dt*p.vel

			PE[n] += 0.5*q*phii

			p.pos[0] %= Lx
			p.pos[1] %= Ly
	"""
	for c in pop:
		for p in c:

			Ei = E(p.pos)
			phii = phi(p.pos)

			q = p.properties['q']
			m = p.properties['m']
			qm = p.properties['qm']

			fraction = 0.5 if n==1 else 1
			inc = dt*fraction*qm*Ei
			vel = p.vel

			KE[n] += 0.5*m*np.dot(vel,vel+inc)

			p.vel += inc
			p.pos += dt*p.vel

			PE[n] += 0.5*q*phii

			p.pos[0] %= Lx
			p.pos[1] %= Ly
	"""

	pop.relocate()

	tPush += t.stop()

fig = plt.figure()
plt.plot(range(1,Nt+1),KE[1:])
plt.plot(range(1,Nt+1),PE[1:])
plt.plot(range(1,Nt+1),PE[1:]+KE[1:])
plt.savefig('energy.png')


TE = KE[1:]+PE[1:]
TEdev = TE - TE[0]
exdev = np.max(np.abs(TEdev))/TE[0]

print "max relative deviation: ", 100*exdev, "%"


print("Finished")

if(show_plot): interactive()
