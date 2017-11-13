import numpy as np

# Filename of mesh (excluding .xml)
fname = "../mesh/2D/langmuir_probe_circle_in_square"

N = 10              # Number of timesteps
dt = 0.15           # Timestep
npc = 4             # Number of particles per cell
cap_factor = 1.5    # Capacitance factor

Rp = 1.                     # Probe radius (in mesh file's units)
Rpd = 1.                    # Probe radius (in debye lengths)
debye = Rp/Rpd              # Debye length
vthe = debye                # Electron thermal velocity
vthi = debye/np.sqrt(1836.) # Ion thermal velocity

Vnorm = debye**2                    # Normalization voltage
Inorm = vthe*np.sqrt(2*np.pi)       # Normalization current
current_collected = -1.556*Inorm    # Current collected
