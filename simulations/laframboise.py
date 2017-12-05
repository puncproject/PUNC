import numpy as np
import dolfin as df
from punc import load_mesh

# Filename of mesh (excluding .xml)
fname = "../mesh/3D/laframboise_sphere_in_cube_res1"
mesh, bnd = load_mesh(fname)

npc = 4             # Number of particles per cell
V = df.assemble(1*df.dx(mesh))
Np = npc*mesh.num_cells()

q_e = 1.6021766208e-19 # C
m_e = 9.10938356e-31 # kg
eps_0 = 8.854187817e-12 # F/m

Rp = 1.
n = 1e10
debye = 1.
w_pe = np.sqrt(n*q_e**2/(eps_0*m_e))

X = 1.0
T = w_pe**(-1)
D = 3
Q = q_e
M = (T*Q)**2 / (eps_0 * X**D)

K = n/(Np/V)
# electron = (-K*q_e,      K*m_e)
# ion      = ( K*q_e, 1836*K*m_e)

electron = (-K*1.0,      K*m_e/M)
ion      = ( K*1.0, 1836*K*m_e/M)

normtype = 'none'



N = 200
dt = 0.1 #*w_pe**(-1)
cap_factor = 1.

vthe = debye #*w_pe
vthi = vthe/np.sqrt(1836.) # Ion thermal velocity

Vnorm = 1./(X/T)**2 * (M/Q)
Inorm = 1./(Q/T)
# Vnorm = 1.
# Inorm = 1.

current_collected = 0

object_method = 'variational'
imposed_potential = 181*Vnorm

eps_0=1
