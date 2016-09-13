from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import time
from punc import *
from WeightedGradient import weighted_gradient_matrix

preview = False
cgspace = True
lattice = False

show_plot = True if preview else False
store_phi = False
set_log_level(WARNING)

np.random.seed(666)

# punc = Punc()
# punc.mesh = RectangleMesh(Point(0,0),Point(Ld),Nc[0],Nc[1])

#==============================================================================
# GENERATE MESH
#------------------------------------------------------------------------------

print "Initializing solver"

Ld = 2*DOLFIN_PI*np.array([1,1])	# Length of domain
Nc = 32*np.array([1,1])				# Number of cells

delta = Ld/Nc						# Stepsizes
Nt = 25 if not preview else 1		# Number of time steps
dt = 0.0251327						# Time step

mesh = RectangleMesh(Point(0,0),Point(Ld),Nc[0],Nc[1])

Npc = 8 if not preview else 128		# Number of particles per cell
Np = mesh.num_cells()*Npc			# Number of particles


if(show_plot): plot(mesh)

punc = Punc(mesh,PeriodicBoundary(Ld))

#==============================================================================
# INITIALIZE PARTICLES
#------------------------------------------------------------------------------

print "Initializing particles"

pop = Population(punc.S,punc.V)

if(lattice):
	punc.pop.addLatticeSine(Np,Ld,0.01)
else:
	punc.pop.addRandomSine(Np,Ld,0.01)


if(False):
	fig = plt.figure()
	pop.scatter(fig)
	fig.suptitle('Initial Particle pos')
	plt.axis([0, Lx, 0, Ly])
	fig.savefig("particles.png")

#==============================================================================
# TIME LOOP
#------------------------------------------------------------------------------

KE = np.zeros(Nt)
PE = np.zeros(Nt)

for n in xrange(1,Nt+1):

	print "Computing time-step %d"%n

	#==========================================================================
	# (POS,Q) -> RHO
	#--------------------------------------------------------------------------

	print "    Accumulating charges"

	punc.distr(np.prod(delta))
	# rho = Function(punc.S)
	# if(not cgspace):
	# 	rho = punc.distrDG0(pop,rho,D)
	# else:
	# 	punc.distrCG1(pop,rho,np.prod(delta))

	if(show_plot): plot(rho)

	#==========================================================================
	# RHO -> PHI
	#--------------------------------------------------------------------------

	print "    Solving potential"

	punc.solve()

	#==========================================================================
	# E -> (VEL,POS)
	#--------------------------------------------------------------------------

	print "    Pushing particles"

	fraction = 0.5 if n==1 else 1.0
	KE[n-1] = punc.accel(dt*fraction)
	PE[n-1] = punc.potEnergy()
	punc.movePeriodic(dt,Ld)

	punc.pop.relocate()

KE[0]=0

fig = plt.figure()
plt.plot(range(Nt),KE)
plt.plot(range(Nt),PE)
plt.plot(range(Nt),PE+KE)
plt.savefig('energy.png')

TE = KE+PE
TEdev = TE - TE[0]
exdev = np.max(np.abs(TEdev))/TE[0]

print "max relative deviation: ", 100*exdev, "%"


print("Finished")

if(show_plot): interactive()
