#!/usr/bin/env python3
import pymatgen as pmg
from hea.tools.properties import Property
import pandas as pd
from optparse import OptionParser
import os.path
import sqlalchemy.types

# molecule = pmg.Molecule.from_file('tests/tests_files/water.xyz')
# structure = mg.Composition(molecule.formula)
comp1 = pmg.Composition("Al0.5NbTaTiV")
comp2 = pmg.Composition("CoCrFeNi")
comp3 = pmg.Composition("AlCrCuFeMnNi")
comp4 = pmg.Composition("AlNi")
comp5 = pmg.Composition("Al0.5CoCrCuFeNiTi0.2")

compositions = ["Al0.5CoCrCuFeNiTi0.2", "Al0.3CoCrFeNi", "Al0.5CrCuFeNi2", "CoCrFeNi", "Al0.5NbTaTiV", "Al0.25NbTaTiV"]
#

props = ["melting_point", "atomic_size_difference", "mixing_entropy", "mixing_enthalpy", "electronegativity", "VEC"]
data = []
for compo in compositions:
    comp = pmg.Composition(compo)
    properties = Property(comp)
    structure_prop = {"Alloys": compo}
    for prop in props:
        structure_prop[prop] = properties.get_property(prop)

    data.append(structure_prop)

data_df = pd.DataFrame(data)
pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 8)
print(data_df.head())
# for key in structure_prop.keys():
#   print('{} : {:.2f}'.format(key, structure_prop[key]))

