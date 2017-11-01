import sys
from punc import *
from dolfin import *

if sys.version_info.major == 2:
	from itertools import izip as zip
	range = xrange

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
