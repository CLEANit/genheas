#!/usr/bin/env python
"""
@Time    : 04 Jan 9:47 a.m. 2021
@Author  : Conrard Tetsassi
@Email   : giresse.feugmo@gmail.com
@File    : optimize.py
@Project : pyHEA
@Software: PyCharm
"""

import os

import torch
import yaml

from hea import generate
from hea.tools.alloysgen import coordination_numbers, properties_list
from hea.tools.feedforward import Feedforward
from hea.tools.log import logger
from hea.train import train_policy

# ========================== Read Parameters  ============================

input_file = 'parameters.yml'
try:
    with open(os.path.join('./', input_file)) as fr:
        parameters = yaml.safe_load(fr)
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

nn_per_policy = parameters['nn_per_policy']
nb_policies = parameters['nb_policies']
nb_worker = parameters['nb_worker']
patience = parameters['patience']

nb_species = len(elements_pool)

nb_structures = parameters['nb_structures']
generation = parameters['nb_generation']
training_size = parameters['cell_size']
supercell_size = parameters['supercell_size']
cube = parameters['cubic']
surface = parameters['surfaces']
# cube = parameters['cubic']
# ==========================  Analyze Parameters  ============================

if not list(oxidation_states.keys()) == elements_pool:
    logger.error('<oxidation_states> should match <elements_pool>')
    raise Exception('<oxidation_states> should match <elements_pool>')
elif sum(oxidation_states.values()) != 0:
    logger.error('sum of oxidation states should be 0')
    raise Exception('sum of oxidation states should be 0')

logger.info('Input parameters have been retrieved')

nn_in_shell1 = coordination_numbers[crystal_structure][0]
nn_in_shell2 = coordination_numbers[crystal_structure][1]
n_neighbours = nn_in_shell1 + 1 + nn_in_shell2

input_size = n_neighbours * len(properties_list)

output_size = len(elements_pool)

# ==============================  Training ===============================

for generation in generations:
    nb_generation = generation
    best_policy = train_policy(
        crystal_structure,
        elements_pool,
        concentrations,
        nb_generation,
        training_size,
        cell_parameters,
        rate=rate,
        alpha=alpha,
        nn_per_policy=nn_per_policy,
        nb_policies=nb_policies,
        nb_worker=nb_worker,
        patience=patience,
        oxidation_states=oxidation_states,
        cubik=False,
        direction=None
    )

    # ============================  Generation  ==========================

    output = os.path.join(str(nb_species) + '_elements', 'model_' + str(nb_generation))

    networks = [Feedforward(input_size, output_size) for i in range(len(best_policy))]
    networks = [network.to(device) for network in networks]

    for j, w in enumerate(best_policy):
        networks[j].l1.weight = torch.nn.Parameter(w)

    generate.generate_structure(
        networks,
        output,
        elements_pool,
        concentrations,
        crystal_structure,
        training_size,
        cell_parameters,
        constraints=False,
    )
