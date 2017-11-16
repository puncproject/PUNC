import numpy as np

# Filename of mesh (excluding .xml)
fname = "../mesh/3D/laframboise_sphere_in_cube_res1"

N = 200             # Number of timesteps
dt = 0.1            # Timestep
npc = 4             # Number of particles per cell
cap_factor = 1.     # Capacitance factor

Rp = 1.                     # Probe radius (in mesh file's units)
Rpd = 1.                    # Probe radius (in debye lengths)
debye = Rp/Rpd              # Debye length
vthe = debye                # Electron thermal velocity
vthi = debye/np.sqrt(1836.) # Ion thermal velocity
electron = 'electron'
ion = 'proton'

Vnorm = debye**2                    # Normalization voltage
Inorm = vthe*np.sqrt(8*np.pi)       # Normalization current
current_collected = -1.987*Inorm    # Current collected

Rp = 1.
Vnorm = Rp**(-2)
Inorm = np.sqrt(8*np.pi)/Rp

object_method = 'capacitance'
imposed_potential = 1.0*Vnorm
