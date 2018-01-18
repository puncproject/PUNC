import numpy as np

# Filename of mesh (excluding .xml)
fname = "../mesh/2D/langmuir_probe_circle_in_square_res2"

N = 1000            # Number of timesteps
dt = 0.15           # Timestep
npc = 4             # Number of particles per cell
cap_factor = 1.5    # Capacitance factor

Rp = 1.                     # Probe radius (in mesh file's units)
# Rpd = 1.                    # Probe radius (in debye lengths)
# debye = Rp/Rpd              # Debye length
# vthe = debye                # Electron thermal velocity
# vthi = debye/np.sqrt(1836.) # Ion thermal velocity
debye = 28.9653
# debye = 1.
vthe = debye
# vthi = debye*np.sqrt(1418.9/1601.3)/np.sqrt(1836.15)
vthi = debye/np.sqrt(1836.15)
# electron = 'electron'
# ion = (1.0, 29164.2) # O+ ions
# ion = 'proton'
# electron = (-1.0, 5.69e-12)
# ion      = (+1.0, 1.05e-8)
# electron = (-1.0, 1e-7)
# ion      = ( 1.0, 1e-7*1836)
electron  = (-1.0, 1.0)
ion       = ( 1.0, 1836)


phys_pfreq = 23.9042e6 # rad/s
phys_debye = 6.51719e-3 # m

X = phys_debye / debye
T = 1. / phys_pfreq
D = 2

# vacuum_permittivity = 8.854187817e-12 # F/m

Vnorm = debye**2                    # Normalization voltage
Inorm = vthe*np.sqrt(2*np.pi)       # Normalization current

q_e = 1.6021766208e-19 # Electron charge [C]

Inorm = 1./8.836210477334834e-06
Inorm = 1./3.481085070434508e-08
Inorm = 1./1.9881465947454995e-09
Inorm = 1./60367456673.42093
Inorm = 1./268299910357476.0
Inorm = 1./13582672.541194193
Inorm = 1.
Inorm = 1./(1e6*q_e/(X*T))
# Vnorm = 1./0.13799
# Inorm = 1./1.407499e23
# Inorm = 1./5888083451116792.0
# Inorm = 1./(21497.22247071758*vacuum_permittivity)
# Inorm = 1./2.2596435764364056e-32
current_collected = -1.556*Inorm    # Current collected

object_method = 'variational'
# imposed_potential = 7.0*Vnorm
imposed_potential = 25.0*Vnorm
