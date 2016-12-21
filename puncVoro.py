from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import time
from punc import *

nDims=3
Ld = 4.0*np.ones(nDims)			# Length of domain
Nc = 4*np.ones(nDims,dtype=int)	# Number of "Rectangles" in mesh

mesh = BoxMesh(Point(0,0,0),Point(Ld),*Nc)

punc = Punc(mesh,PeriodicBoundary(Ld))
punc.compVolume(Ld)

#plot(punc.dv)


print("Finished")
