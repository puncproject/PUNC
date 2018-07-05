#!/usr/bin/env python

# Usage:
#    ./cellvolume.py mydolfinmesh.msh 1.234
#
#    where 1.234 is the Debye length

from __future__ import division, print_function
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys

mesh = Mesh(sys.argv[1])

volumes = np.array([c.volume() for c in cells(mesh)])
volumes.sort()

debye = float(sys.argv[2])
valid = np.sum(volumes<debye**3)
valid2 = np.sum(volumes<(3.4*debye)**3)
total = len(volumes)

print("Cell volume range: %f - %f"%(volumes[0],volumes[-1]))
print("Debye cube volume: %f"%debye**3)
print("%d of %d cells (%2.1f%%) have volumes less than a Debye cube"%(valid,total,100.*valid/total))
print("%d of %d cells (%2.1f%%) have stable volumes"%(valid2,total,100.*valid2/total))
print("Unstable volumes leads to numerical heating until stability is reached.")
