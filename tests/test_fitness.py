#!/usr/bin/env python
"""
@Time    : Jan 18 3:23 p.m. 2021
@Author  : Conrard TETSASSI
@Email   : giresse.feugmo@gmail.com
@File    : test_fitness.py
@Project : genheas
@Software: PyCharm
"""
from genheas import generate
from genheas.tools.alloysgen import AlloysGen
from genheas.tools.alloysgen import coordination_numbers
from genheas.tools.evolution import NnEa


def test_fitness():
    """read cif file"""
    # cif = ase.io.read('tests_files/Fe_mp-13_primitive.cif')

    assert True
