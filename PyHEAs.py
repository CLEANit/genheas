#!/usr/bin/env python3
import pymatgen as pmg
from Tools import Properties
from Tools import miedema
from optparse import OptionParser
import os.path








molecule = pmg.Molecule.from_file('tests/tests_files/water.xyz')
#structure = mg.Composition(molecule.formula)
comp = pmg.Composition("Al0.25NbTaTiV")
comp2 = pmg.Composition("CoCrFeNi")
comp3 = pmg.Composition("AlCrCuFeMnNi")
comp4 = pmg.Composition("FeNi")
properties = Properties.Property(comp3)


# miedema_prop = miedema.Miedema.get({'Fe': 1.0, 'Ni': 1.0})
# miedema_prop2 = miedema.Miedema.get("FeNi")
# miedema_prop3 = miedema.Miedema.get(pmg.Composition("FeNi"))


props = ["melting_point", "atomic_size_difference", "mixing_entropy", "mixing_enthalpy", "electronegativity", "VEC"]
structure_prop = {}
for prop in props:
    structure_prop[prop] = properties.get_property(prop)

for item in structure_prop.items():
    print(item)






