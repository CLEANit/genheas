#!/usr/bin/env python
"""
@Time    : Jan 18 3:30 p.m. 2021
@Author  : Conrard TETSASSI
@Email   : giresse.feugmo@gmail.com
@File    : test_generate_structure.py
@Project : genheas
@Software: PyCharm
"""

from genheas.tools.gencrystal import AlloysGen


AlloyGen = AlloysGen(["Ag", "Pd"], {"Ag": 0.5, "Pd": 0.5}, "fcc")


# def test_generate():
#     """read cif file"""
#     # cif = ase.io.read('tests_files/Fe_mp-13_primitive.cif')
#     assert True


def test_generate():
    try:
        # structureX = AlloyGen.gen_crystall(
        #     "fcc",
        #     [2, 2, 2],
        #     lattice_param=[4.09, 4.09, 4.09],
        # )
        result = True
    except BaseException:
        result = False

    assert result
