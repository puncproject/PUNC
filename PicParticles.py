import LagrangianParticles as lp

class Particle(lp.Particle):
    __slots__ = ['velocity'] # Inherits other slots

	def __init__(self, x, v):
		lp.Particle.__init__(self, x)
		self.velocity = v

	def send(self, x, v):
		
