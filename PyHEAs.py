#!/usr/bin/env python3
import pymatgen as mg

from optparse import OptionParser
import os.path



si = mg.Element("Si")
si.atomic_mass

inputfile = 'tests/tests_files/water.xyz'

#structure = mg.Structure.from_str(open(inputfile).read(), fmt="xyz")
molecule = mg.Molecule.from_file('tests/tests_files/water.xyz')
structure = mg.Composition(molecule.formula)
comp = mg.Composition("Fe2O3")
#print(structure.cart_coords)
print(structure.to_data_dict
#structure2 = mg.Structure.from_file('tests/tests_files/Fe_mp-13_primitive.cif')
#print(structure.get_atomic_fraction("H"))

structure.

