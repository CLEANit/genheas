#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 04 Jan 9:47 a.m. 2021
@Author  : Conrard Tetsassi
@Email   : giresse.feugmo@gmail.com
@File    : hea_gen.py
@Project : pyHEA
@Software: PyCharm
"""

import glob
import os

import numpy as np
import torch
import yaml

from hea import generate
from hea.tools.alloysgen import coordination_numbers, properties_list
from hea.tools.feedforward import Feedforward
from hea.tools.log import logger
from hea.train import train_policy

# ========================== Read Parameters  ============================
try:
    with open(os.path.join('./', "parameters.yml"), "r") as fr:
        parameters = yaml.safe_load(fr)
        logger.info('Parameters have been retrieved')
except Exception as err:
    logger.error(f'{err}')
    raise Exception(f'{err}')

generations = parameters['nb_generation']
crystal_structure = parameters['crystalstructure']
cell_parameters = parameters['cell_parameters']
elements_pool = parameters['elements_pool']
concentrations = parameters['concentrations']
oxidation_states = parameters['oxidation_states']
rate = parameters['rate']
alpha = parameters['alpha']
device = parameters['device']

nb_species = len(elements_pool)

nb_structures = parameters['nb_structures']
generation = parameters['generation']
size = parameters['size']

# ==========================  Analyze Parameters  ============================

if not list(oxidation_states.keys()) == elements_pool:
    raise Exception('<oxidation_states> should match <elements_pool>')
elif sum(oxidation_states.values()) != 0:
    raise Exception("sum of oxidation states should be 0")

training_files = glob.glob('training_data/{}/clusters*'.format(crystal_structure))
training_sizes = [x.split('/')[-1].split('_')[-1] for x in training_files]
training_sizes.sort(reverse=True)

composition = np.array(list(concentrations.values()))
composition = composition / np.min(composition)
min_nb_atoms = int(np.sum(composition))

training_size = next(x for x in training_sizes if int(x) >= min_nb_atoms)

if min_nb_atoms < 8:
    training_size = 8

NN_in_shell1 = coordination_numbers[crystal_structure][0]
NN_in_shell2 = coordination_numbers[crystal_structure][1]
NNeighbours = NN_in_shell1 + 1 + NN_in_shell2

input_size = NNeighbours * len(properties_list)

output_size = len(elements_pool)

# ==============================  Training ===============================


for generation in generations:
    nb_generation = generation
    # Feedforward
    best_policy = train_policy(crystal_structure, elements_pool, concentrations, nb_generation,
                               training_size, oxidation_states=oxidation_states)

    # ============================  Generation  ==========================

    output = os.path.join(str(nb_species) + '_elemetns', 'model_' + str(nb_generation))
    if generation:
        model = Feedforward(input_size, output_size)
        model.to(device)
        model.l1.weight = torch.nn.parameter.Parameter(best_policy)
        generate.generate_structure(model, output, elements_pool, concentrations, crystal_structure, size,
                                    cell_parameters)
