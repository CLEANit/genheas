#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : Jan 18 3:30 p.m. 2021
@Author  : Conrard TETSASSI
@Email   : giresse.feugmo@gmail.com
@File    : test_generate_structure.py
@Project : pyHEA
@Software: PyCharm
"""

from hea import generate
from hea.tools.nn_ga_model import NnGa
from hea.tools.alloysgen import (
    AlloysGen,
    coordination_numbers,
    properties_list,
)

GaNn = NnGa()
AlloyGen = AlloysGen(['Ag', 'Pd'], {'Ag': 0.5, 'Pd': 0.5}, 'fcc')


def test_generate():
    try:
        structureX = AlloyGen.gen_random_structure('fcc', [2, 2, 2], {'Ag': 4, 'Pd': 4},
                                                   lattice_param=[4.09, 4.09, 4.09])
        result = True
    except:
        result = False

    assert result == True
