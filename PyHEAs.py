#!/usr/bin/env python3
import pymatgen as mg
from Tools import Properties
from optparse import OptionParser
import os.path






molecule = mg.Molecule.from_file('tests/tests_files/water.xyz')
#structure = mg.Composition(molecule.formula)
comp = mg.Composition("Al0.25NbTaTiV")
#comp = mg.Composition("CoCrFeNi")

properties = Properties.Property(comp)

props = ["melting_point", "atomic_size_difference", "mixing_entropy", "electronegativity", "VEC"]
structure_prop = {}
for prop in props:
    structure_prop[prop] = properties.get_property(prop)

print(structure_prop)




