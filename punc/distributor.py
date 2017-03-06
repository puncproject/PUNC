# __authors__ = ('Sigvald Marholm <sigvaldm@fys.uio.no>')
# __date__ = '2017-02-22'
# __copyright__ = 'Copyright (C) 2017' + __authors__
# __license__  = 'GNU Lesser GPL version 3 or any later version'

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
import pyvoro

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
