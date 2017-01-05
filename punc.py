# __authors__ = ('Sigvald Marholm <sigvaldm@fys.uio.no>',
#				 'Mikael Mortensen <mikaem@math.uio.no>',
#				 'Miroslav Kuchta <mirok@math.uio.no>')
# __date__ = '2014-19-11'
# __copyright__ = 'Copyright (C) 2011' + __authors__
# __license__  = 'GNU Lesser GPL version 3 or any later version'
#
# Based on fenicstools/LagrangianParticles
#

from dolfin import *
import numpy as np
import copy
from mpi4py import MPI as pyMPI
from collections import defaultdict
import pyvoro
from subprocess import call	
import time
import sys

# Python 2/3 compatibility notes:
#
# Python 2 is old, but unfortunately still the most used. To make code as
# compatible with both as possible:
#
# Write print("...") instead of print "..."
# Always use range and zip instead of xrange and izip which are obsolete.
# The below test takes care of getting their performance in python 2

if sys.version_info.major == 2:
	from itertools import izip as zip
	range = xrange

# Disable printing
__DEBUG__ = False

comm = pyMPI.COMM_WORLD

# collisions tests return this value or -1 if there is no collision
__UINT32_MAX__ = np.iinfo('uint32').max

class Langmuir(object):
	def __init__(self):

		# Default values
		self.Ld = 2*DOLFIN_PI*np.array([1,1])
		self.Nc = 32*np.array([1,1])
		self.Nt = 25
		self.dt = 0.1*2*DOLFIN_PI/self.Nt
		self.Npc = 8
		self.amp = 0.01

		# Set up with default settings. Use set-functions to override.
		self.setMesh()
		self.setTime()

	def reinit(self):
		self.setMesh()
		self.setTime()

	def setMesh(self, Nc=None ,Ld=None):
		if Nc != None: self.Nc = Nc
		if Ld != None: self.Ld = Ld
		self.mesh = RectangleMesh(Point(0,0),Point(self.Ld),*self.Nc)
		self.punc = Punc(self.mesh,self.Ld,PeriodicBoundary(self.Ld))
		self.setPop() # finer grid => more particles => update population

	def setTime(self, dt=None, Nt=None):
		if dt != None: self.dt = dt
		if Nt != None: self.Nt = Nt
		self.KE = np.zeros(self.Nt)
		self.PE = np.zeros(self.Nt)
		self.TE = np.zeros(self.Nt)

	def setPop(self, Npc=None, amp=None):
		if Npc != None: self.Npc = Npc
		if amp != None: self.amp = amp
		self.Np = self.mesh.num_cells()*self.Npc
		self.punc.pop.addSine(self.Np,self.Ld,self.amp)

	def run(self):
		for n in xrange(1,self.Nt+1):

			print("    Computing time-step %d"%n)

			print("        Accumulating charges")
			self.punc.distr(np.prod(self.Ld/self.Nc))

			print("        Solving potential")
			self.punc.solve()

			print("        Pushing particles")

			fraction = 0.5 if n==1 else 1.0
			self.KE[n-1] = self.punc.accel(self.dt*fraction)
			self.PE[n-1] = self.punc.potEnergy()
			self.punc.movePeriodic(self.dt,self.Ld)

		self.KE[0] = 0 # assuming 0 initial velocity
		self.TE = self.KE+self.PE
		relError = (self.TE-self.TE[0])/self.TE[0]
		return np.max(np.abs(relError))

class Punc(object):

	def __init__(self,mesh,Ld,constr):

		assert has_linear_algebra_backend("PETSc")

		parameters["linear_algebra_backend"] = "PETSc"
		set_log_level(WARNING)
		np.random.seed(666)

		#
		# FUNCTION SPACE (discontinuous scalar, scalar, vector)
		#
		self.mesh = mesh
		self.D =       FunctionSpace(mesh, 'DG', 0, constrained_domain=constr)
		self.S =       FunctionSpace(mesh, 'CG', 1, constrained_domain=constr)
		self.V = VectorFunctionSpace(mesh, 'CG', 1, constrained_domain=constr)

		#
		# INITIALIZE SOLVER
		#
		phi = TrialFunction(self.S)
		phi_ = TestFunction(self.S)

		a = inner(nabla_grad(phi), nabla_grad(phi_))*dx
		A = assemble(a)

		self.solver = PETScKrylovSolver("cg")
		self.solver.set_operator(A)

		self.phi_ = phi_
		self.phi = Function(self.S)
		self.rho = Function(self.S)
		self.E = Function(self.V)

		null_vec = Vector(self.phi.vector())
		self.S.dofmap().set(null_vec, 1.0)
		null_vec *= 1.0/null_vec.norm("l2")

		self.null_space = VectorSpaceBasis([null_vec])
		as_backend_type(A).set_nullspace(self.null_space)

		#
		# POPULATION
		#
		self.pop = Population(self.S,self.V)

		#
		# COMPUTE VOLUME OF VORONOI CELLS
		#
		self.compVolume(Ld) # Add Npc to increase efficiency

	def compVolume(self, Ld, Npc=8, bnd="Periodic"):

		assert bnd=="Periodic"

		mesh = self.S.mesh()
		vertices = mesh.coordinates()
		dofs = vertex_to_dof_map(self.S)

		# Remove those on upper bound (admittedly inefficient)
		i = 0
#		while i<len(doVePairs):
#			if any([near(a,b) for a,b in zip(doVePairs[i][1],Ld)]):
#				doVePairs.pop(i)
#			else:
#				i = i+1
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
		limits = np.zeros([nDims,2])# - 0.001
		limits[:,1] = Ld            # + 0.001

		# ~5 particles per block yields better performance. Thus correctness of
		# Npc is not actually important.
		nParticles = Npc*self.mesh.num_cells()
		nBlocks = nParticles/5.0
		nBlocksPerDim = int(nBlocks**(1/nDims)) # integer feels safer
		blockSize = np.prod(Ld)**(1/nDims)/nBlocksPerDim

		if nDims==2:
			voronoi = pyvoro.compute_2d_voronoi(vertices,limits,blockSize,periodic=[True]*2)
		if nDims==3:
			voronoi = pyvoro.compute_voronoi(vertices,limits,blockSize,periodic=[True]*3)		

		dvArr = [vcell['volume'] for vcell in voronoi]
		dvInvArr = [vcell['volume']**(-1) for vcell in voronoi]

		self.dv = Function(self.S)
		self.dvInv = Function(self.S)

		# There must be some way to avoid this loop
		for i in range(len(dvArr)):
			self.dv.vector()[i] = dvArr[i]
			self.dvInv.vector()[i] = dvInvArr[i]

		# self.dv is now a FEniCS function which on the vertices of the FEM mesh
		# equals the volume of the Voronoi cells created from those vertices.
		# It's meaningless to evaluate self.dv in-between vertices. Since it's
		# cheaper to multiply by its inverse we've computed self.dvInv too.
		# We actually don't need self.dv except for debugging.

	def solve(self):

		L = self.rho*self.phi_*dx
		b = assemble(L)
		self.null_space.orthogonalize(b);

		self.solver.solve(self.phi.vector(), b)

		self.E = project(-grad(self.phi), self.V)

	def accel(self,dt):

		KE = 0.0

		pop = self.pop

		for c in cells(self.mesh):

			self.E.restrict(pop.Vcoefficients,
							pop.Velement,
							c,
							c.get_vertex_coordinates(),
							c)

			for p in pop[c.index()]:
				pop.Velement.evaluate_basis_all(	pop.Vbasis_matrix,
														p.pos,
														c.get_vertex_coordinates(),
														c.orientation())

				Ei = np.dot(pop.Vcoefficients, pop.Vbasis_matrix)[:]

				m = p.properties['m']
				qm = p.properties['qm']

				inc = dt*qm*Ei
				vel = p.vel

				KE += 0.5*m*np.dot(vel,vel+inc)

				p.vel += inc

		return KE

	def movePeriodic(self,dt,L):
		for cell in self.pop:
			for particle in cell:
				particle.pos += dt*particle.vel
				particle.pos %= L
		self.pop.relocate()

	def potEnergy(self):

		PE = 0
		pop = self.pop

		for c in cells(self.mesh):
			self.phi.restrict(	pop.Scoefficients,
								pop.Selement,
								c,
								c.get_vertex_coordinates(),
								c)

			for p in pop[c.index()]:
				pop.Selement.evaluate_basis_all(	pop.Sbasis_matrix,
													p.pos,
													c.get_vertex_coordinates(),
													c.orientation())

				phii = np.dot(pop.Scoefficients, pop.Sbasis_matrix)[:]

				q = p.properties['q']
				PE += 0.5*q*phii

		return PE

	def distr(self,da,method='CG1voro'):
		assert method in ['CG1','DG0','CG1voro']
		# da is the area/volume assumed to be constant, used only by CG1

		pop = self.pop
		self.rho = Function(self.S)	# Sets to zero

		if method == 'DG0':
			rhoD = Function(D)

			for c in cells(self.mesh):
				cindex = c.index()
				dofindex = self.D.dofmap().cell_dofs(cindex)[0]
				cellcharge = 0
				da = c.volume()
				for particle in pop[cindex]:
					cellcharge += particle.properties['q']/da
				rhoD.vector()[dofindex] = cellcharge

			self.rho = project(rhoD,self.S)

		if method == 'CG1':

			for c in cells(self.mesh):
				cindex = c.index()
				dofindex = self.S.dofmap().cell_dofs(cindex)

				accum = np.zeros(3)
				for p in pop[cindex]:

					pop.Selement.evaluate_basis_all(	pop.Sbasis_matrix,
														p.pos,
														c.get_vertex_coordinates(),
														c.orientation())

					q=p.properties['q']
					accum += (q/da)*pop.Sbasis_matrix.T[0]

				self.rho.vector()[dofindex] += accum

		if method == 'CG1voro':

			for c in cells(self.mesh):
				cindex = c.index()
				dofindex = self.S.dofmap().cell_dofs(cindex)

				accum = np.zeros(3)
				for p in pop[cindex]:

					pop.Selement.evaluate_basis_all(	pop.Sbasis_matrix,
														p.pos,
														c.get_vertex_coordinates(),
														c.orientation())

					q=p.properties['q']
					accum += q*pop.Sbasis_matrix.T[0]

				self.rho.vector()[dofindex] += accum
			
			# Divide by area/volume
			self.rho.vector()[:] *= self.dvInv.vector()

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

class Particle:
	__slots__ = ['pos', 'vel', 'properties']		# changed
	'Lagrangian particle with pos and some other passive properties.'
	def __init__(self, x, v=0):
		if(isinstance(x,Particle)): return x
		self.pos = np.array(x)
		if(v==0): v=np.zeros(len(x))
		self.vel = np.array(v)
		self.properties = {}

	def send(self, dest):
		'Send particle to dest.'
		comm.Send(self.pos, dest=dest)
		comm.Send(self.vel, dest=dest)					# new
		comm.send(self.properties, dest=dest)

	def recv(self, source):
		'Receive info of a new particle sent from source.'
		comm.Recv(self.pos, source=source)
		comm.Recv(self.vel, source=source)				# new
		self.properties = comm.recv(source=source)

class Population(list):
	'Particles moved by the vel field in V.'
	def __init__(self, S, V):
		self.__debug = __DEBUG__

		self.S = S
		self.mesh = S.mesh()
		self.mesh.init(2, 2)  # Cell-cell connectivity for neighbors of cell
		self.tree = self.mesh.bounding_box_tree()  # Tree for isection comput.

		for cell in cells(self.mesh):
			self.append(list())

		# Allocate some variables used to look up the vel
		# vel is computed as U_i*basis_i where i is the dimension of
		# element function space, U are coefficients and basis_i are element
		# function space basis functions. For interpolation in cell it is
		# advantageous to compute the resctriction once for cell and only
		# update basis_i(x) depending on x, i.e. particle where we make
		# interpolation. This updaea mounts to computing the basis matrix
		self.dim = self.mesh.topology().dim()

		self.Velement = V.dolfin_element()
		self.Vnum_tensor_entries = 1
		for i in range(self.Velement.value_rank()):
			self.Vnum_tensor_entries *= self.Velement.value_dimension(i)
		# For VectorFunctionSpace CG1 this is 3
		self.Vcoefficients = np.zeros(self.Velement.space_dimension())
		# For VectorFunctionSpace CG1 this is 3x3
		self.Vbasis_matrix = np.zeros((self.Velement.space_dimension(),
									  self.Vnum_tensor_entries))

		self.Selement = S.dolfin_element()
		self.Snum_tensor_entries = 1
		for i in range(self.Selement.value_rank()):
			self.Snum_tensor_entries *= self.Selement.value_dimension(i)
		# For VectorFunctionSpace CG1 this is 3
		self.Scoefficients = np.zeros(self.Selement.space_dimension())
		# For VectorFunctionSpace CG1 this is 3x3
		self.Sbasis_matrix = np.zeros((self.Selement.space_dimension(),
									  self.Snum_tensor_entries))

		# Allocate a dictionary to hold all particles


		# Allocate some MPI stuff
		self.num_processes = comm.Get_size()
		self.myrank = comm.Get_rank()
		self.all_processes = list(range(self.num_processes))
		self.other_processes = list(range(self.num_processes))
		self.other_processes.remove(self.myrank)
		self.my_escaped_particles = np.zeros(1, dtype='I')
		self.tot_escaped_particles = np.zeros(self.num_processes, dtype='I')
		# Dummy particle for receiving/sending at [0, 0, ...]
		self.particle0 = Particle(np.zeros(self.mesh.geometry().dim()))

	def addParticles(self, list_of_particles, properties_d=None):
		'''Add particles and search for their home on all processors.
		   Note that list_of_particles must be same on all processes. Further
		   every len(properties[property]) must equal len(list_of_particles).
		'''
		if properties_d is not None:
			n = len(list_of_particles)
			assert all(len(sub_list) == n
					   for sub_list in properties_d.itervalues())
			# Dictionary that will be used to feed properties of single
			# particles
			properties = properties_d.keys()
			particle_properties = dict((key, 0) for key in properties)

			has_properties = True
		else:
			has_properties = False

		my_found = np.zeros(len(list_of_particles), 'I')
		all_found = np.zeros(len(list_of_particles), 'I')
		for i, particle in enumerate(list_of_particles):
			c = self.locate(particle)
			if not (c == -1 or c == __UINT32_MAX__):
				my_found[i] = True

				self[c].append(Particle(particle))

				if has_properties:
					# Get values of properties for this particle
					for key in properties:
						particle_properties[key] = properties_d[key][i]
					self[c][-1].properties.update(particle_properties)

		# All particles must be found on some process
		comm.Reduce(my_found, all_found, root=0)

		if self.myrank == 0:
			missing = np.where(all_found == 0)[0]
			n_missing = len(missing)

			assert n_missing == 0,\
				'%d particles are not located in mesh' % n_missing

			# Print particle info
			if self.__debug:
				for i in missing:
					print 'Missing', list_of_particles[i].pos

				n_duplicit = len(np.where(all_found > 1)[0])
				print 'There are %d duplicit particles' % n_duplicit

	def addSine(self,Np,Ld,amp,method='random'):
		assert method in ['random','lattice']

		q = np.array([-1., 1.])
		m = np.array([1., 1836.])

		multiplicity = (np.prod(Ld)/np.prod(Np))*m[0]/(q[0]**2)
		q *= multiplicity
		m *= multiplicity
		qm = q/m

		if method=='random':
			# Electrons
			pos = RandomRectangle(Point(0,0),Point(Ld)).generate(Np)
			pos[:,0] += amp*np.sin(pos[:,0])
			qTemp = q[0]*np.ones(len(pos))
			mTemp = m[0]*np.ones(len(pos))
			qmTemp = qm[0]*np.ones(len(pos))
			self.addParticles(pos,{'q':qTemp,'qm':qmTemp,'m':mTemp})

			# Ions
			pos = RandomRectangle(Point(0,0),Point(Ld)).generate(Np)
			qTemp = q[1]*np.ones(len(pos))
			mTemp = m[1]*np.ones(len(pos))
			qmTemp = qm[1]*np.ones(len(pos))
			self.addParticles(pos,{'q':qTemp,'qm':qmTemp,'m':mTemp})

		if method=='lattice':

			x = np.arange(0,Ld[0],Ld[0]/Np[0])
			y = np.arange(0,Ld[1],Ld[1]/Np[1])
			x += amp*np.sin(x)
			xcart = np.tile(x,Np[1])
			ycart = np.repeat(y,Np[0])
			pos = np.c_[xcart,ycart]
			qTemp = q[0]*np.ones(len(pos))
			mTemp = m[0]*np.ones(len(pos))
			qmTemp = qm[0]*np.ones(len(pos))
			self.addParticles(pos,{'q':qTemp,'qm':qmTemp,'m':mTemp})

			x = np.arange(0,Ld[0],Ld[0]/Np[0])
			y = np.arange(0,Ld[1],Ld[1]/Np[1])
			xcart = np.tile(x,Np[1])
			ycart = np.repeat(y,Np[0])
			pos = np.c_[xcart,ycart]
			qTemp = q[1]*np.ones(len(pos))
			mTemp = m[1]*np.ones(len(pos))
			qmTemp = qm[1]*np.ones(len(pos))
			self.addParticles(pos,{'q':qTemp,'qm':qmTemp,'m':mTemp})

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
					for neighbor in cells(dfCell):
						if neighbor.contains(point):
							new_cell_id = neighbor.index()
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
			point = Point(*particle.pos)
			return self.tree.compute_first_entity_collision(point)
		else:
			return self.locate(Particle(particle))

	def scatter(self, fig, skip=1):
		'Scatter plot of all particles on process 0'
		import matplotlib.colors as colors
		import matplotlib.cm as cmx

		ax = fig.gca()

		p_map = self.particle_map
		all_particles = np.zeros(self.num_processes, dtype='I')
		my_particles = p_map.total_number_of_particles()
		# Root learns about count of particles on all processes
		comm.Gather(np.array([my_particles], 'I'), all_particles, root=0)

		# Slaves should send to master
		if self.myrank > 0:
			for cwp in p_map.itervalues():
				for p in cwp.particles:
					p.send(0)
		else:
			# Receive on master
			received = defaultdict(list)
			received[0] = [copy.copy(p.pos)
						   for cwp in p_map.itervalues()
						   for p in cwp.particles]
			for proc in self.other_processes:
				# Receive all_particles[proc]
				for j in range(all_particles[proc]):
					self.particle0.recv(proc)
					received[proc].append(copy.copy(self.particle0.pos))

			cmap = cmx.get_cmap('jet')
			cnorm = colors.Normalize(vmin=0, vmax=self.num_processes)
			scalarMap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)

			for proc in received:
				# Plot only if there is something to plot
				particles = received[proc]
				if len(particles) > 0:
					xy = np.array(particles)

					ax.scatter(xy[::skip, 0], xy[::skip, 1],
							   label='%d' % proc,
							   c=scalarMap.to_rgba(proc),
							   edgecolor='none')
			ax.legend(loc='best')
			ax.axis([0, 1, 0, 1])

	def bar(self, fig):
		'Bar plot of particle distribution.'
		ax = fig.gca()

		p_map = self.particle_map
		all_particles = np.zeros(self.num_processes, dtype='I')
		my_particles = p_map.total_number_of_particles()
		# Root learns about count of particles on all processes
		comm.Gather(np.array([my_particles], 'I'), all_particles, root=0)

		if self.myrank == 0 and self.num_processes > 1:
			ax.bar(np.array(self.all_processes)-0.25, all_particles, 0.5)
			ax.set_xlabel('proc')
			ax.set_ylabel('number of particles')
			ax.set_xlim(-0.25, max(self.all_processes)+0.25)
			return np.sum(all_particles)
		else:
			return None

# Simple initializers for particle poss

from math import pi, sqrt
from itertools import product

comm = pyMPI.COMM_WORLD

class RandomGenerator(object):
	'''
	Fill object by random points.
	'''
	def __init__(self, domain, rule):
		'''
		Domain specifies bounding box for the shape and is used to generate
		points. The rule filter points of inside the bounding box that are
		axctually inside the shape.
		'''
		assert isinstance(domain, list)
		self.domain = domain
		self.rule = rule
		self.dim = len(domain)
		self.rank = comm.Get_rank()

	def generate(self, N, method='full'):
		'Genererate points.'
		assert (isinstance(N,int) and method == 'full') or len(N) == self.dim
		assert method in ['full', 'tensor']

		if self.rank == 0:
			# Generate random points for all coordinates
			if method == 'full':
				n_points = np.product(N)
				points = np.random.rand(n_points, self.dim)
				for i, (a, b) in enumerate(self.domain):
					points[:, i] = a + points[:, i]*(b-a)
			# Create points by tensor product of intervals
			else:
				# Values from [0, 1) used to create points between
				# a, b - boundary
				# points in each of the directiosn
				shifts_i = np.array([np.random.rand(n) for n in N])
				# Create candidates for each directions
				points_i = (a+shifts_i[i]*(b-a)
							for i, (a, b) in enumerate(self.domain))
				# Cartesian product of directions yield n-d points
				points = (np.array(point) for point in product(*points_i))


			# Use rule to see which points are inside
			points_inside = np.array(filter(self.rule, points))
		else:
			points_inside = None

		points_inside = comm.bcast(points_inside, root=0)

		return points_inside


class RandomRectangle(RandomGenerator):
	def __init__(self, ll, ur):
		ax, ay = ll.x(), ll.y()
		bx, by = ur.x(), ur.y()
		assert ax < bx and ay < by
		RandomGenerator.__init__(self, [[ax, bx], [ay, by]], lambda x: True)


class RandomCircle(RandomGenerator):
	def __init__(self, center, radius):
		assert radius > 0
		domain = [[center[0]-radius, center[0]+radius],
				  [center[1]-radius, center[1]+radius]]
		RandomGenerator.__init__(self, domain,
								 lambda x: sqrt((x[0]-center[0])**2 +
												(x[1]-center[1])**2) < radius
								 )
