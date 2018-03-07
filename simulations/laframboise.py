import numpy as np
import dolfin as df
from punc import *
import scipy.constants as constants

# Filename of mesh (excluding .xml)
fname = "../mesh/3D/laframboise_sphere_in_cube_res1"

# Get the mesh
mesh, bnd = load_mesh(fname)
ext_bnd_id, int_bnd_ids = get_mesh_ids(bnd)
ext_bnd = ExteriorBoundaries(bnd, ext_bnd_id)

npc = 4             # Number of particles per cell
V = df.assemble(1*df.dx(mesh))
Np = npc*mesh.num_cells()

me   = constants.value('electron mass')
mp   = constants.value('proton mass')
e    = constants.value('elementary charge')
eps0 = constants.value('electric constant')
kB   = constants.value('Boltzmann constant')

ne      = 1e10
# Te    = 1000
# debye = np.sqrt(eps0*kB*Te/(e**2 * ne))
debye   = 1.0
Te      = (e*debye)**2*ne/(eps0*kB)
wpe     = np.sqrt(ne*e**2/(eps0*me))
vthe    = debye*wpe
vthi    = vthe/np.sqrt(1836)
Rp      = 1*debye
X       = Rp

Vlam    = kB*Te/e
Ilam    = -e*ne*Rp**2*np.sqrt(8*np.pi*kB*Te/me)
Iexp    = 1.987*Ilam
print("Laframboise voltage:  %e"%Vlam)
print("Laframboise current:  %e"%Ilam)
print("Expected current:     %e"%Iexp)

species = SpeciesList(mesh, ext_bnd, X)
species.append(-e, me, ne, vthe, npc=npc)
species.append(e, mp, ne, vthi, npc=npc)

Inorm  = species.Q/species.T
Vnorm  = (species.M/species.Q)*(species.X/species.T)**2
# Inorm  = 1.
# Vnorm  = 1.
Inorm /= np.abs(Ilam)
Vnorm /= Vlam

N          = 400
dt         = 0.05#*wpe**(-1)
cap_factor = 1.

current_collected = Iexp/(species.Q/species.T)

object_method = 'variational'
imposed_potential = 1.0/Vnorm

eps0=1
