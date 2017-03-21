import numpy as np

epsilon_0 = 1.              # Permittivity of vacuum
mu_0 = 1.                   # Permeability of vacuum
T_e = 1.                    # Temperature - electrons
T_i = 1.                    # Temperature - ions
kB = 1.                     # Boltzmann's constant
e = 1.                      # Elementary charge
Z = 1                       # Atomic number
m_e = 1.                    # particle mass - electron
m_i = 1836.15267389         # particle mass - ion

alpha_e = np.sqrt(kB*T_e/m_e) # Boltzmann factor
alpha_i = np.sqrt(kB*T_i/m_i) # Boltzmann factor

q_e = -e         # Electric charge - electron
q_i = Z*e        # Electric charge - ions
