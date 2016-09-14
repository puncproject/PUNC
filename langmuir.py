from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import time
from punc import *
from WeightedGradient import weighted_gradient_matrix

preview = False

#==============================================================================
# GENERATE MESH
#------------------------------------------------------------------------------

print "Initializing solver"

Ld = 2*DOLFIN_PI*np.array([1,1])	# Length of domain
Nc = 32*np.array([1,1])				# Number of 'rectangles' in mesh

Nt = 25 if not preview else 1		# Number of time steps
dt = 0.0251327						# Time step

mesh = RectangleMesh(Point(0,0),Point(Ld),*Nc)

Npc = 8 if not preview else 128		# Number of particles per (triangular) cell
Np = mesh.num_cells()*Npc			# Number of particles


punc = Punc(mesh,PeriodicBoundary(Ld))

#==============================================================================
# INITIALIZE PARTICLES
#------------------------------------------------------------------------------

print "Initializing particles"
punc.pop.addSine(Np,Ld,0.01)

#==============================================================================
# TIME LOOP
#------------------------------------------------------------------------------

KE = np.zeros(Nt)
PE = np.zeros(Nt)

for n in xrange(1,Nt+1):

	print "Computing time-step %d"%n

	print "    Accumulating charges"
	punc.distr(np.prod(Ld/Nc))

	print "    Solving potential"
	punc.solve()

	print "    Pushing particles"

	fraction = 0.5 if n==1 else 1.0
	KE[n-1] = punc.accel(dt*fraction)
	PE[n-1] = punc.potEnergy()
	punc.movePeriodic(dt,Ld)

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
