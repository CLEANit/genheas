#!/usr/bin/env python3
import pymatgen as mg
from Tools import Properties
from optparse import OptionParser
import os.path






molecule = mg.Molecule.from_file('tests/tests_files/water.xyz')
#structure = mg.Composition(molecule.formula)
comp = mg.Composition("Al0.25NbTaTiV")
comp2 = mg.Composition("CoCrFeNi")
comp3 = mg.Composition("CuNi")

properties = Properties.Property(comp3)

props = ["melting_point", "atomic_size_difference", "mixing_entropy", "electronegativity", "VEC"]
structure_prop = {}
for prop in props:
    structure_prop[prop] = properties.get_property(prop)

print(structure_prop)




