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

E = VectorFunctionSpace(...)
rho = Function(...)
phi = Function(...)


fsolver = PeriodicPoissonSolver()
pop = Population()
acc = Accelerator()
mov = Mover()
dist = Distributer()

# Mark objects

# Adding electrons
vd, vth = ..., ...
q, m, N = stdParticleParameters(mesh,Npc,qm,mme)
xs = randomPoints(lambda x: ..., Ld, N)
vs = maxwellian(vd, vth)
pop.addParticles(xs,vs,q,m)

# Adding ions
vd, vth = ..., ...
q, m, N = stdParticleParameters(mesh,Npc,qm,mme)
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

class PeriodicPoissonSolver(object):

	def solve(self,rho,E):

		L = self.rho*self.phi_*dx
		b = assemble(L)

		self.null_space.orthogonalize(b);

		self.solver.solve(self.phi.vector(), b)

		self.E = project(-grad(self.phi), self.CG1V)

class PoissonSolver2(object):

	def __init__(self, mesh, Ld, bnd):

		assert(bnd in ['periodic','dirichlet'])

		#
		# FUNCTION SPACES AND BOUNDARY CONDITIONS
		#
		self.bnd = bnd
		if(bnd=='periodic'): constr = PeriodicBoundary(Ld)
		else: constr = None

		self.mesh = mesh
		self.DG0  =       FunctionSpace(mesh,'DG',0,constrained_domain=constr)
		self.CG1  =       FunctionSpace(mesh,'CG',1,constrained_domain=constr)
		self.CG1V = VectorFunctionSpace(mesh,'CG',1,constrained_domain=constr)

		if(bnd=='dirichlet'):
			bc = DirichletBC(self.CG1,Constant(0),extBnd)

		#
		# INITIALIZE SOLVER
		#
		phi = TrialFunction(self.CG1)
		phi_ = TestFunction(self.CG1)

		a = inner(nabla_grad(phi), nabla_grad(phi_))*dx
		A = assemble(a)

		self.solver = PETScKrylovSolver("cg")
		self.solver.set_operator(A)

		self.phi_ = phi_
		self.phi = Function(self.CG1)
		self.rho = Function(self.CG1)
		self.E = Function(self.CG1V)

		if(bnd=='periodic'):
			# Remove null-vector from solution space
			null_vec = Vector(self.phi.vector())
			self.CG1.dofmap().set(null_vec, 1.0)
			null_vec *= 1.0/null_vec.norm("l2")
			self.null_space = VectorSpaceBasis([null_vec])
			as_backend_type(A).set_nullspace(self.null_space)

	def solve(self):

		L = self.rho*self.phi_*dx
		b = assemble(L)

		if(self.bnd=='periodic'):
			self.null_space.orthogonalize(b);

		self.solver.solve(self.phi.vector(), b)

		self.E = project(-grad(self.phi), self.CG1V)

class Accel:

	def __init__(self, mesh, bnd):
		pass

class Distr:

	def __init__(self, mesh, Ld, bnd="periodic"):

		assert bnd=="periodic"

		self.mesh = mesh
		self.CG1 = FunctionSpace(mesh, 'CG', 1)

		vertices = mesh.coordinates()
		dofs = vertex_to_dof_map(self.CG1)

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

		#self.dv = Function(self.CG1)
		self.dvInv = Function(self.CG1)

		# There must be some way to avoid this loop
		for i in range(len(dvInvArr)):
			#self.dv.vector()[i] = dvArr[i]
			self.dvInv.vector()[i] = dvInvArr[i]

		# self.dv is now a FEniCS function which on the vertices of the FEM mesh
		# equals the volume of the Voronoi cells created from those vertices.
		# It's meaningless to evaluate self.dv in-between vertices. Since it's
		# cheaper to multiply by its inverse we've computed self.dvInv too.
		# We actually don't need self.dv except for debugging.

	def distr(self,pop,rho):
		# rho assumed to be CG1

		element = self.CG1.dolfin_element()
		sDim = element.space_dimension() # Number of nodes per element
		basisMatrix = np.zeros((sDim,1))

		rho.vector()[:] = 0

		for cell in cells(self.mesh):
			cellindex = cell.index()
			dofindex = self.CG1.dofmap().cell_dofs(cellindex)

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

class Population(list):

	def __init__(self, CG1, Ld):
		self.CG1 = CG1
		self.mesh = CG1.mesh()
		self.Ld = Ld

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

		self.compVoronoi(Ld)

	def compVoronoi(self, Ld, bnd="periodic"):

		assert bnd=="periodic"

		mesh = self.CG1.mesh()
		vertices = mesh.coordinates()
		dofs = vertex_to_dof_map(self.CG1)

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

		#self.dv = Function(self.CG1)
		self.dvInv = Function(self.CG1)

		# There must be some way to avoid this loop
		for i in range(len(dvInvArr)):
			#self.dv.vector()[i] = dvArr[i]
			self.dvInv.vector()[i] = dvInvArr[i]

		# self.dv is now a FEniCS function which on the vertices of the FEM mesh
		# equals the volume of the Voronoi cells created from those vertices.
		# It's meaningless to evaluate self.dv in-between vertices. Since it's
		# cheaper to multiply by its inverse we've computed self.dvInv too.
		# We actually don't need self.dv except for debugging.

	def addParticles(self, xs, vs, qs, ms):
		# Adds particles to the population and locates them on their home
		# processor. xs is a list/array of position vectors. vs, qs and ms may
		# be lists/arrays of velocity vectors, charges, and masses,
		# respectively, or they may be only a single velocity vector, mass
		# and/or charge if all particles should have the same value.

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
		# Relocate particles on cells and processors
		# Map such that map[old_cell] = [(new_cell, particle_id), ...]
		# Ie new destination of particles formerly in old_cell
		new_cell_map = defaultdict(list)
		for dfCell in cells(self.mesh):
			cindex = dfCell.index()
			cell = self[cindex]
			for i, particle in enumerate(cell):
				point = Point(*particle.pos)
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

	def distr(self,rho):
		# rho assumed to be CG1

		element = self.CG1.dolfin_element()
		sDim = element.space_dimension() # Number of nodes per element
		basisMatrix = np.zeros((sDim,1))

		rho.vector()[:] = 0

		for cell in cells(self.mesh):
			cellindex = cell.index()
			dofindex = self.CG1.dofmap().cell_dofs(cellindex)

			accum = np.zeros(sDim)
			for particle in self[cellindex]:

				element.evaluate_basis_all(	basisMatrix,
											particle.x,
											cell.get_vertex_coordinates(),
											cell.orientation())

				accum += particle.q * basisMatrix.T[0]

			rho.vector()[dofindex] += accum

		# Divide by volume of Voronoi cell
		rho.vector()[:] *= self.dvInv.vector()

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
	# Returns N maxwellian velocity vectors with drift and thermal velocities
	# vd and vth, respectively. If you have a list/array of position vectors
	# pos an equivalen way to create them would be
	# 	np.random.normal(vd, vth, pos.shape)
	return np.random.normal(vd, vth, (N,len(vd)) )
