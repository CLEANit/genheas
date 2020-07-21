
import numpy

from scipy.constants import codata

import scipy


#Molar gas constant

R = codata.value("molar gas constant") # J/K/mol.
#scipy.constants.R

# speed of light in atomic units

c = codata.value('elementary charge')
G = codata.value("Newtonian constant of gravitation")
mu_0 = codata.value('mag. constant')
epsilon_0 = codata.value('electric constant')
cvel = codata.value('inverse fine-structure constant')
cvel_ms = codata.value('speed of light in vacuum')
a = codata.value('fine-structure constant')

# conversion from bohr to Angstrom
Bohr_in_Angstrom = codata.value('Bohr radius') *1.0e10
Bohr_in_Meter = codata.value('Bohr radius')

# Conversion mass from amu to au of mass (electron mass)
Avogadro = codata.value('Avogadro constant')

amu_in_kg = codata.value('atomic mass constant')
amu_in_electronmass = 1.0/codata.value('electron mass in u')


Hartree_in_Joule = codata.value('atomic unit of energy')
Hartree_in_eV = codata.value('Hartree energy in eV')
Hartree_in_invcm = codata.value('hartree-inverse meter relationship') /100.0

invcm_in_Hartree = codata.value('inverse meter-hartree relationship') *100.0

au_in_Debye = 2.541746

try:
    au_in_electricpolarizability = codata.value('atomic unit of electric polarizability')#C2 m2 /J
except KeyError:#Mistake in the name in the previous version of scipy
    au_in_electricpolarizability = codata.value('atomic unit of electric polarizablity')#C2 m2 /J

try:
    au_in_firsthyperpolarizability = codata.value('atomic unit of 1st hyperpolarizability')
except KeyError:#Mistake in the name in the previous version of scipy
    au_in_firsthyperpolarizability = codata.value('atomic unit of 1st hyperpolarizablity')

Boltzman_constant = codata.value('Boltzmann constant')

hbar = codata.value('Planck constant over 2 pi')
h = codata.value('Planck constant')

q = codata.value('elementary charge')
electronmass = codata.value('electron mass')
epsilon0 = codata.value('electric constant')




def epsilon(i, j, k):
    """
    levi-civita symbol
    """
    return(j-i) *(k-i) *(k-j) /2


def conv_freq_in_angvel_au(freq):
    #w = 2 pi c nubar(m-1) = 200 pi c nubar(cm-1)
#    w = 200.0*numpy.pi*cvel*freq*Bohr_in_Meter
    w = freq/Hartree_in_invcm
    return w

def boltz(freq, T=298.15):
    r"""
    (1-exp(-\hbar freqmode/kT)) **-1
    Param freq:one or list of frequencies in Hartree
    """
    fact = 1.0-numpy.exp(-freq*Hartree_in_Joule/(Boltzman_constant*T))
    return 1.0/fact

