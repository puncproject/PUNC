# __authors__ = ('Sigvald Marholm <sigvaldm@fys.uio.no>')
# __date__ = '2017-02-22'
# __copyright__ = 'Copyright (C) 2017' + __authors__
# __license__  = 'GNU Lesser GPL version 3 or any later version'
#
# Loosely based on fenicstools/LagrangianParticles by Mikeal Mortensen and
# Miroslav Kuchta. Released under same license.

# Imports important python 3 behaviour to ensure correct operation and
# performance in python 2
from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
	from itertools import izip as zip
	range = xrange

#import dolfin as df
from dolfin import *
import numpy as np
from mpi4py import MPI as pyMPI
from collections import defaultdict
from itertools import count
import pyvoro

comm = pyMPI.COMM_WORLD

# collisions tests return this value or -1 if there is no collision
__UINT32_MAX__ = np.iinfo('uint32').max

"""

Ld = ...
mesh = ...

V = FunctionSpace(mesh, 'CG', 1, constr=constr)
Vv = VectorFunctionSpace(mesh, 'CG', 1, constr=constr)

E = Function(Vv)
rho = Function(V)
phi = Function(V)


fsolver = PeriodicPoissonSolver(V)
pop = Population()
acc = Accelerator()
mov = Mover()
dist = Distributer()

# Mark objects

# Adding electrons
vd, vth = ..., ...
q, m, N = stdSpecie(mesh, Ld, -1, 1, 8)
xs = randomPoints(lambda x: ..., Ld, N)
vs = maxwellian(vd, vth)
pop.addParticles(xs,vs,q,m)

# Adding ions
vd, vth = ..., ...
q, m, N = stdSpecie(mesh, Ld, -1, 1, 8)
xs = randomPoints(lambda x: ..., Ld, N)
vs = maxwellian(vd, vth)
pop.addParticles(xs,vs,q,m)

# x, v given in n=0

KE0 = kinEnergy(pop)

Nt = ...
for n in range(1,Nt):
	dist.dist(pop,RHO)
	bcs = boundaryConds(...)
	PHI = fsolver.solve(RHO)		# PHI = fsolver.dirichlet_solver(RHO,bcs)
	E = gradient(PHI)
	PE[n-1] = potEnergy(RHO,PHI)
	KE[n-1] = acc.acc(pop,E,(1-0.5*(n==1))*dt)	# v now at step n-0.5
	mov.move(pop)								# x now at step n
	objCurrent(...)
	inject(...)
	delete(...)

KE[0] = KE0

"""

def stdSpecie(mesh, Ld, q, m, N, q0=-1.0, m0=1.0, wp0=1.0, count='per cell'):
	"""
	Returns standard normalized particle parameters to use for a specie. The
	inputs q and m are the charge and mass of the specie in elementary charges
	and electron masses, respectively. N is the number of particles per cell
	unless count='total' in which case it is the total number of simulation
	particles in the domain. For instance to create 8 ions per cell (having
	mass of 1836 electrons and 1 positive elementary charge):

		q, m, N = stdSpecie(mesh, Ld, 1, 1836, 8)

	The returned values are the normalized charge q, mass m and total number of
	simulation particles N. The normalization is such that the angular electron
	plasma frequency will be 1.

	Alternatively, the reference charge q0 and mass m0 which should yield a
	angular plasma frequency of 1, or for that sake any other frequency wp0,
	could be set through the kwargs. For example, in this case the ions will
	have a plasma frequency of 0.1:

		q, m, N = stdSpecie(mesh, Ld, 1, 1836, 8, q0=1, m0=1836, wp0=0.1)
	"""

	assert count in ['per cell','total']

	if count=='total':
		Np = N
	else:
		Np = N*mesh.num_cells()

	mul = (np.prod(Ld)/np.prod(Np))*(wp0**2)*m0/(q0**2)
	return q*mul, m*mul, Np

class PoissonSolver:

	def __init__(self, V):

		self.solver = PETScKrylovSolver('gmres', 'hypre_amg')
		self.solver.parameters['absolute_tolerance'] = 1e-14
		self.solver.parameters['relative_tolerance'] = 1e-12
		self.solver.parameters['maximum_iterations'] = 1000

		self.V = V

		phi = TrialFunction(V)
		phi_ = TestFunction(V)

		a = inner(nabla_grad(phi), nabla_grad(phi_))*dx
		A = assemble(a)

		self.solver.set_operator(A)
		self.phi_ = phi_

		phi = Function(V)
		null_vec = Vector(phi.vector())
		V.dofmap().set(null_vec, 1.0)
		null_vec *= 1.0/null_vec.norm("l2")

		self.null_space = VectorSpaceBasis([null_vec])
		as_backend_type(A).set_nullspace(self.null_space)

	def solve(self, rho):

		L = rho*self.phi_*dx
		b = assemble(L)
		self.null_space.orthogonalize(b);

		phi = Function(self.V)
		self.solver.solve(phi.vector(), b)

		return phi

class Accel:

	def __init__(self, mesh, bnd):
		pass

class Distributor:

	def __init__(self, V, Ld, bnd="periodic"):

		assert bnd=="periodic"

		self.V = V
		self.mesh = V.mesh()

		vertices = self.mesh.coordinates()
		dofs = vertex_to_dof_map(self.V)

		# Remove those on upper bound (admittedly inefficient)
		i = 0
		while i<len(vertices):
			if any([near(a,b) for a,b in zip(vertices[i],list(Ld))]):
				vertices = np.delete(vertices,[i],axis=0)
				dofs = np.delete(dofs,[i],axis=0)
			else:
				i = i+1

		# Sort vertices to appear in the order FEniCS wants them for the DOFs
		sortIndices = np.argsort(dofs)
		vertices = vertices[sortIndices]

		nDims = len(Ld)
		limits = np.zeros([nDims,2])
		limits[:,1] = Ld

		# ~5 particles (vertices) per block yields better performance.
		nParticles = self.mesh.num_vertices()
		nBlocks = nParticles/5.0
		nBlocksPerDim = int(nBlocks**(1/nDims)) # integer feels safer
		blockSize = np.prod(Ld)**(1/nDims)/nBlocksPerDim

		if nParticles>24000:
			print("Warning: The pyvoro library often experience problems with many particles. This despite the fact that voro++ should be well suited for big problems.")

		if nDims==1:
			error("1D voronoi not implemented yet")
		if nDims==2:
			voronoi = pyvoro.compute_2d_voronoi(vertices,limits,blockSize,periodic=[True]*2)
		if nDims==3:
			voronoi = pyvoro.compute_voronoi(vertices,limits,blockSize,periodic=[True]*3)

		#dvArr = [vcell['volume'] for vcell in voronoi]
		dvInvArr = [vcell['volume']**(-1) for vcell in voronoi]

		#self.dv = Function(self.V)
		self.dvInv = Function(self.V)

		# There must be some way to avoid this loop
		for i in range(len(dvInvArr)):
			#self.dv.vector()[i] = dvArr[i]
			self.dvInv.vector()[i] = dvInvArr[i]

		# self.dv is now a FEniCS function which on the vertices of the FEM mesh
		# equals the volume of the Voronoi cells created from those vertices.
		# It's meaningless to evaluate self.dv in-between vertices. Since it's
		# cheaper to multiply by its inverse we've computed self.dvInv too.
		# We actually don't need self.dv except for debugging.

	def distr(self,pop):
		# rho assumed to be CG1

		element = self.V.dolfin_element()
		sDim = element.space_dimension() # Number of nodes per element
		basisMatrix = np.zeros((sDim,1))

		rho = Function(self.V)

		for cell in cells(self.mesh):
			cellindex = cell.index()
			dofindex = self.V.dofmap().cell_dofs(cellindex)

			accum = np.zeros(sDim)
			for particle in pop[cellindex]:

				element.evaluate_basis_all(	basisMatrix,
											particle.x,
											cell.get_vertex_coordinates(),
											cell.orientation())

				accum += particle.q * basisMatrix.T[0]

			rho.vector()[dofindex] += accum

		# Divide by volume of Voronoi cell
		rho.vector()[:] *= self.dvInv.vector()

		return rho

class Population(list):

	def __init__(self, mesh):
		self.mesh = mesh

		# Allocate a list of particles for each cell
		for cell in cells(self.mesh):
			self.append(list())

		# Create a list of sets of neighbors for each cell
		self.tDim = self.mesh.topology().dim()
		self.gDim = self.mesh.geometry().dim()
		self.mesh.init(0, self.tDim)
		self.tree = self.mesh.bounding_box_tree()
		self.neighbors = list()
		for cell in cells(self.mesh):
			neigh = sum([vertex.entities(self.tDim).tolist() for vertex in vertices(cell)], [])
			neigh = set(neigh) - set([cell.index()])
			self.neighbors.append(neigh)

		# Allocate some MPI stuff
		self.num_processes = comm.Get_size()
		self.myrank = comm.Get_rank()
		self.all_processes = list(range(self.num_processes))
		self.other_processes = list(range(self.num_processes))
		self.other_processes.remove(self.myrank)
		self.my_escaped_particles = np.zeros(1, dtype='I')
		self.tot_escaped_particles = np.zeros(self.num_processes, dtype='I')
		# Dummy particle for receiving/sending at [0, 0, ...]
		vZero = np.zeros(self.gDim)
		self.particle0 = Particle(vZero,vZero,1,1)

	def addParticles(self, xs, vs=None, qs=None, ms=None):
		"""
		Adds particles to the population and locates them on their home
		processor. xs is a list/array of position vectors. vs, qs and ms may
		be lists/arrays of velocity vectors, charges, and masses,
		respectively, or they may be only a single velocity vector, mass
		and/or charge if all particles should have the same value.
		"""

		if vs==None or qs==None or ms==None:
			assert isinstance(xs,list)
			if len(xs)==0:
				return
			assert isinstance(xs[0],Particle)
			ps = xs
			xs = [p.x for p in ps]
			vs = [p.v for p in ps]
			qs = [p.q for p in ps]
			ms = [p.m for p in ps]
			addParticles(xs, vs, qs, ms)
			return

		# Expand input to lists/arrays if necessary
		if len(np.array(vs).shape)==1: vs = np.tile(vs,(len(xs),1))
		if not isinstance(qs, (np.ndarray,list)): qs *= np.ones(len(xs))
		if not isinstance(ms, (np.ndarray,list)): ms *= np.ones(len(xs))

		# Keep track of which particles are located locally and globally
		my_found = np.zeros(len(xs), np.int)
		all_found = np.zeros(len(xs), np.int)

		for i, x, v, q, m in zip(count(), xs, vs, qs, ms):
			cell = self.locate(x)
			if not (cell == -1 or cell == __UINT32_MAX__):
				my_found[i] = True
				self[cell].append(Particle(x, v, q, m))

		# All particles must be found on some process
		comm.Reduce(my_found, all_found, root=0)

		if self.myrank == 0:
			nMissing = len(np.where(all_found == 0)[0])
			assert nMissing==0,'%d particles are not located in mesh'%nMissing

	def relocate(self):
		"""
		Relocate particles on cells and processors
		map such that map[old_cell] = [(new_cell, particle_id), ...]
		i.e. new destination of particles formerly in old_cell
		"""
		new_cell_map = defaultdict(list)
		for dfCell in cells(self.mesh):
			cindex = dfCell.index()
			cell = self[cindex]
			for i, particle in enumerate(cell):
				point = Point(*particle.x)
				# Search only if particle moved outside original cell
				if not dfCell.contains(point):
					found = False
					# Check neighbor cells
					for neighbor in self.neighbors[dfCell.index()]:
						if Cell(self.mesh,neighbor).contains(point):
							new_cell_id = neighbor
							found = True
							break
					# Do a completely new search if not found by now
					if not found:
						new_cell_id = self.locate(particle)
					# Record to map
					new_cell_map[dfCell.index()].append((new_cell_id, i))

		# Rebuild locally the particles that end up on the process. Some
		# have cell_id == -1, i.e. they are on other process
		list_of_escaped_particles = []
		for old_cell_id, new_data in new_cell_map.iteritems():
			# We iterate in reverse becasue normal order would remove some
			# particle this shifts the whole list!
			for (new_cell_id, i) in sorted(new_data,
										   key=lambda t: t[1],
										   reverse=True):
#				particle = p_map.pop(old_cell_id, i)

				particle = self[old_cell_id][i]
				# Delete particle in old cell, fill gap by last element
				if not i==len(self[old_cell_id])-1:
					self[old_cell_id][i] = self[old_cell_id].pop()
				else:
					self[old_cell_id].pop()

				if new_cell_id == -1 or new_cell_id == __UINT32_MAX__ :
					list_of_escaped_particles.append(particle)
				else:
#					p_map += self.mesh, new_cell_id, particle
					self[new_cell_id].append(particle)

		# Create a list of how many particles escapes from each processor
		self.my_escaped_particles[0] = len(list_of_escaped_particles)
		# Make all processes aware of the number of escapees
		comm.Allgather(self.my_escaped_particles, self.tot_escaped_particles)

		# Send particles to root
		if self.myrank != 0:
			for particle in list_of_escaped_particles:
				particle.send(0)

		# Receive the particles escaping from other processors
		if self.myrank == 0:
			for proc in self.other_processes:
				for i in range(self.tot_escaped_particles[proc]):
					self.particle0.recv(proc)
					list_of_escaped_particles.append(copy.deepcopy(self.particle0))

		# Put all travelling particles on all processes, then perform new search
		travelling_particles = comm.bcast(list_of_escaped_particles, root=0)
		self.addParticles(travelling_particles)

	def total_number_of_particles(self):
		'Return number of particles in total and on process.'
		num_p = self.particle_map.total_number_of_particles()
		tot_p = comm.allreduce(num_p)
		return (tot_p, num_p)

	def locate(self, particle):
		'Find mesh cell that contains particle.'
		assert isinstance(particle, (Particle, np.ndarray))
		if isinstance(particle, Particle):
			# Convert particle to point
			point = Point(*particle.x)
		else:
			point = Point(*particle)
		return self.tree.compute_first_entity_collision(point)

class Particle:
	__slots__ = ['x', 'v', 'q', 'm']
	def __init__(self, x, v, q, m):
		assert(q!=0 and m!=0)
		self.x = np.array(x)	# Position vector
		self.v = np.array(v)	# Velocity vector
		self.q = q				# Charge
		self.m = m				# Mass

	def send(self, dest):
		comm.Send(self.x, dest=dest)
		comm.Send(self.v, dest=dest)
		comm.Send(self.q, dest=dest)
		comm.Send(self.m, dest=dest)

	def recv(self, source):
		comm.Recv(self.x, source=source)
		comm.Recv(self.v, source=source)
		comm.Recv(self.q, source=source)
		comm.Recv(self.m, source=source)

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

def randomPoints(pdf, Ld, N, pdfMax=1):
	# Creates an array of N points randomly distributed according to pdf in a
	# domain size given by Ld (and starting in the origin) using the Monte
	# Carlo method. The pdf is assumed to have a max-value of 1 unless
	# otherwise specified. Useful for creating arrays of position/velocity
	# vectors of various distributions.

	Ld = np.array(Ld) # In case it's a list
	dim = len(Ld)

	assert not isinstance(pdf(np.ones(dim)),(list,np.ndarray)) ,\
		"pdf returns non-scalar value"

	points = np.array([]).reshape(0,dim)

	while len(points)<N:

		# Creates missing points
		n = N-len(points)
		newPoints = np.random.rand(n,dim+1) # Last value to be compared with pdf
		newPoints *= np.append(Ld,pdfMax)	# Stretch axes

		# Only keep points below pdf
		newPoints = [x[0:dim] for x in newPoints if x[dim]<pdf(x[0:dim])]
		newPoints = np.array(newPoints).reshape(-1,dim)
		points = np.concatenate([points,newPoints])

	return points

def maxwellian(vd, vth, N):
	"""
	Returns N maxwellian velocity vectors with drift and thermal velocity
	vectors vd and vth, respectively, in an (N,d) array where d is the number
	of geometric dimensions. If you have an (N,d) array xs (e.g. position
	vectors) the following call would also be valid:

		maxwellian(vd, vth, xs.shape)

	or even

		np.random.normal(vd, vth, xs.shape)

	where np stands for numpy. However, np.random.normal fails when one of the
	elements in the vector vth is zero. maxwellian replaces zeroes with the
	machine epsilon.

	Either vd or vth may be a scalar, or both if N is a shape-tuple.
	"""

	if isinstance(N,int):
		d = max(len(vd),len(vth))
	else:
		(N,d) = N

	# Replace zeroes in vth for machine epsilon
	if isinstance(vth,(float,int)): vth = [vth]
	vth = np.array(vth,dtype=float)
	indices = np.where(vth==0.0)[0]
	vth[indices] = np.finfo(float).eps

	return np.random.normal(vd, vth, (N,d))

def initLangmuir(pop, Ld, vd, vth, A, mode, Npc):
	"""
	Initializes population pop with particles corresponding to electron-ion
	plasma oscillations with amplitude A in density, thermal velocity vth[0]
	and vth[1] for electrons and ions, respectively, and drift velocity vector
	vd. mesh and Ld is the mesh and it's size, while Npc is the number of
	simulation particles per cell per specie. Normalization is such that
	angular electron plasma frequency is one. mode is the number of half
	periods per domain length in the x-direction.
	"""

	mesh = pop.mesh

	# Adding electrons
	q, m, N = stdSpecie(mesh, Ld, -1, 1, Npc)
	pdf = lambda x: 1+A*np.sin(mode*2*np.pi*x[0]/Ld[0])
	xs = randomPoints(pdf, Ld, N, pdfMax=1+A)
	vs = maxwellian(vd, vth[0], xs.shape)
	pop.addParticles(xs,vs,q,m)

	# Adding ions
	q, m, N = stdSpecie(mesh, Ld, 1, 1836, Npc)
	pdf = lambda x: 1
	xs = randomPoints(pdf, Ld, N)
	vs = maxwellian(vd, vth[1], xs.shape)
	pop.addParticles(xs,vs,q,m)

def EField(phi):
	V = phi.ufl_function_space()
	mesh = V.mesh()
	degree = V.ufl_element().degree()
	constr = V.constrained_domain
	W = VectorFunctionSpace(mesh, 'CG', degree, constrained_domain=constr)
	return project(-grad(phi), W)

def accel(pop, E, dt):

	W = E.function_space()
	mesh = W.mesh()
	element = W.dolfin_element()
	sDim = element.space_dimension()  # Number of nodes per element
	vDim = element.value_dimension(0) # Number of values per node (=geom. dim.)
	basisMatrix = np.zeros((sDim,vDim))
	coefficients = np.zeros(sDim)

	KE = 0.0
	for cell in cells(mesh):

		E.restrict(	coefficients,
					element,
					cell,
					cell.get_vertex_coordinates(),
					cell)

		for particle in pop[cell.index()]:
			element.evaluate_basis_all(	basisMatrix,
										particle.x,
										cell.get_vertex_coordinates(),
										cell.orientation())

			Ei = np.dot(coefficients, basisMatrix)[:]

			m = particle.m
			q = particle.q

			inc = dt*(q/m)*Ei
			vel = particle.v

			KE += 0.5*m*np.dot(vel,vel+inc)

			particle.v += inc

	return KE

def movePeriodic(pop, Ld, dt):
	for cell in pop:
		for particle in cell:
			particle.x += dt*particle.v
			particle.x %= Ld
	pop.relocate()

def kineticEnergy(pop):
	"""
	Computes kinetic energy at current velocity time step.
	Useful for the first (zeroth) time step before velocity has between
	advanced half a timestep. To get velocity between two velocity time
	steps (e.g. at integer steps after the start-up) use accel() return.
	"""
	KE = 0
	for cell in pop:
		for particle in cell:
			m = particle.m
			v = particle.v
			KE += 0.5*m*np.dot(v,v)
	return KE

def potentialEnergy(pop,phi):

	PE = 0

	V = phi.function_space()
	element = V.dolfin_element()
	sDim = element.space_dimension()  # Number of nodes per element
	basisMatrix = np.zeros((sDim,1))
	coefficients = np.zeros(sDim)

	for cell in cells(pop.mesh):
		phi.restrict(	coefficients,
						element,
						cell,
						cell.get_vertex_coordinates(),
						cell)

		for particle in pop[cell.index()]:
			element.evaluate_basis_all(	basisMatrix,
										particle.x,
										cell.get_vertex_coordinates(),
										cell.orientation())

			phii = np.dot(coefficients, basisMatrix)[:]

			q = particle.q
			PE += 0.5*q*phii

	return PE
