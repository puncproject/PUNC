# __authors__ = ('Sigvald Marholm <sigvaldm@fys.uio.no>',
#				 'Mikael Mortensen <mikaem@math.uio.no>',
#				 'Miroslav Kuchta <mirok@math.uio.no>')
# __date__ = '2014-19-11'
# __copyright__ = 'Copyright (C) 2011' + __authors__
# __license__  = 'GNU Lesser GPL version 3 or any later version'
#
# Based on fenicstools/LagrangianParticles
#
'''
This module contains functionality for Lagrangian tracking of particles with
DOLFIN
'''

import dolfin as df
import numpy as np
import copy
from mpi4py import MPI as pyMPI
from collections import defaultdict

# Disable printing
__DEBUG__ = False

comm = pyMPI.COMM_WORLD

# collisions tests return this value or -1 if there is no collision
__UINT32_MAX__ = np.iinfo('uint32').max

class Particle:
	__slots__ = ['pos', 'vel', 'properties']		# changed
	'Lagrangian particle with pos and some other passive properties.'
	def __init__(self, x, v=0):
		if(isinstance(x,Particle)): return x
		self.pos = x
		if(v==0): v=np.zeros(len(x))						# new
		self.vel = v									# new
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

		for cell in df.cells(self.mesh):
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
		self.all_processes = range(self.num_processes)
		self.other_processes = range(self.num_processes)
		self.other_processes.remove(self.myrank)
		self.my_escaped_particles = np.zeros(1, dtype='I')
		self.tot_escaped_particles = np.zeros(self.num_processes, dtype='I')
		# Dummy particle for receiving/sending at [0, 0, ...]
		self.particle0 = Particle(np.zeros(self.mesh.geometry().dim()))
		
	"""
	def __iter__(self):
		'''Iterate over all particles.'''
		for cwp in self.particle_map.itervalues():
			for particle in cwp.particles:
				yield particle
	"""
	"""
	def __iter__(self):
		for cell in list.__iter__(self):
			for particle in cell:
				yield particle
	"""
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

	def step(self, u, dt):
		'Move particles by forward Euler x += u*dt'
		start = df.Timer('shift')
		for cwp in self.particle_map.itervalues():
			# Restrict once per cell
			u.restrict(self.coefficients,
					   self.Velement,
					   cwp,
					   cwp.get_vertex_coordinates(),
					   cwp)
			for particle in cwp.particles:
				x = particle.pos
				# Compute vel at pos x
				self.Velement.evaluate_basis_all(self.basis_matrix,
												x,
												cwp.get_vertex_coordinates(),
												cwp.orientation())
				x[:] = x[:] + dt*np.dot(self.coefficients, self.basis_matrix)[:]
		# Recompute the map
		stop_shift = start.stop()
		start =df.Timer('relocate')
		info = self.relocate()
		stop_reloc = start.stop()
		# We return computation time per process
		return (stop_shift, stop_reloc)

	def relocate(self):
		# Relocate particles on cells and processors
		# Map such that map[old_cell] = [(new_cell, particle_id), ...]
		# Ie new destination of particles formerly in old_cell
		new_cell_map = defaultdict(list)
		for dfCell in df.cells(self.mesh):
			cindex = dfCell.index()
			cell = self[cindex]
			for i, particle in enumerate(cell):
				point = df.Point(*particle.pos)
				# Search only if particle moved outside original cell
				if not dfCell.contains(point):
					found = False
					# Check neighbor cells
					for neighbor in df.cells(dfCell):
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
			point = df.Point(*particle.pos)
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
		assert len(N) == self.dim
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
