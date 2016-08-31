import LagrangianParticles as lp
from mpi4py import MPI as pyMPI
import numpy as np
import dolfin as df
from collections import defaultdict

comm = pyMPI.COMM_WORLD
__UINT32_MAX__ = np.iinfo('uint32').max
# Disable printing
__DEBUG__ = False

class Particle:
    __slots__ = ['pos', 'vel']

    def __init__(self, pos, vel):
        self.pos = pos	# position
        self.vel = vel	# velocity

    def send(self, dest):
        comm.Send(self.pos, dest=dest)
        comm.Send(self.vel, dest=dest)

    def recv(self, source):
        comm.Recv(self.pos, source=source)
		comm.Recv(self.vel, source=source)

class Population(list):

	def __init__(self, V):
		self.V = V
		self.mesh = V.mesh()
		for c in df.cells(self.mesh):
			self.append(list())
