from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import time

import sys
#sys.path.append('../src')
#from punc import *

from punc import *

if sys.version_info.major == 2:
	range = xrange

preview = False

#==============================================================================
# GENERATE MESH
#------------------------------------------------------------------------------

print "Initializing solver"

Ld = 6.28*np.array([1,1])	# Length of domain
Nc = 32*np.array([1,1])				# Number of 'rectangles' in mesh

Nt = 25 if not preview else 1		# Number of time steps
dt = 0.251327						# Time step

#mesh = RectangleMesh(Point(0,0),Point(Ld),*Nc)
mesh = Mesh("mesh/nonuniform.xml")
if preview:
	wiz = plot(mesh)
	wiz.write_png("mesh")

Npc = 8 if not preview else 2048	# Number of particles per (triangular) cell
Np = mesh.num_cells()*Npc			# Number of particles

# Average cell area
daAvg = np.prod(Ld)/mesh.num_cells()


punc = Punc(mesh,Ld,PeriodicBoundary(Ld))

#==============================================================================
# INITIALIZE PARTICLES
#------------------------------------------------------------------------------

print "Initializing particles"
punc.pop.addSine(Np,Ld,0.1)

#==============================================================================
# TIME LOOP
#------------------------------------------------------------------------------

KE = np.zeros(Nt)
PE = np.zeros(Nt)

for n in range(1,Nt+1):

	print("Computing time-step %d"%n)

	print("    Accumulating charges")
	punc.distr(daAvg,"CG1voro")
	if preview:
		wiz = plot(punc.rho)
		wiz.write_png("rho")

	print("    Solving potential")
	punc.solve()
	if preview: plot(punc.phi)

	if not preview:
		print("    Pushing particles")

		fraction = 0.5 if n==1 else 1.0
		KE[n-1] = punc.accel(dt*fraction)
		PE[n-1] = punc.potEnergy()
		punc.movePeriodic(dt,Ld)

KE[0]=0

if(preview): interactive()

fig = plt.figure()
plt.plot(range(Nt),KE)
plt.plot(range(Nt),PE)
plt.plot(range(Nt),PE+KE)
plt.savefig('energy.png')

TE = KE+PE
TEdev = TE - TE[0]
exdev = np.max(np.abs(TEdev))/TE[0]

print("Maximum relative energy deviation: %f"%exdev)


print("Finished")
