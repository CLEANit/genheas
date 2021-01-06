#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Fri Oct 16 23:27:41 2020

@author: Conrard Tetsassi
"""

import os
import pathlib
import pickle
import shutil
import time

import matplotlib.pyplot as plt
import torch
import yaml
from ase.io import write
from ase.lattice.cubic import BodyCenteredCubic, FaceCenteredCubic
from PIL import Image

from hea.tools.alloysgen import (AlloysGen, coordination_numbers,
                                 properties_list)
from hea.tools.feedforward import Feedforward
from hea.tools.log import logger
from hea.tools.nn_ga_model import NnGa


# import pymatgen as pmg
# from ase.build import bulk
# from itertools import cycle
# import numpy as np
# from pymatgen.io.ase import AseAtomsAdaptor

def read_model(path, element_pool, device):
    NN_in_shell1 = coordination_numbers[crystal_structure][0]
    NN_in_shell2 = coordination_numbers[crystal_structure][1]
    NNeighbours = NN_in_shell1 + 1 + NN_in_shell2

    input_size = NNeighbours * len(properties_list)

    output_size = len(element_pool)

    my_model = Feedforward(input_size, output_size)
    my_model.to(device)

    # weights = read_policy()
    # model.l1.weight = torch.nn.parameter.Parameter(weights)

    try:
        my_model.load_state_dict(torch.load(path))
    except Exception as err:
        raise Exception(f'{err}')
    return my_model


def read_policy(file):
    weights = pickle.load(open(file, 'r'))
    return weights


def write_to_cif(name, configuration):
    write('{}/structure.cif'.format(name), configuration)


def write_to_png(name, configuration):
    write('{}/structure.png'.format(name), configuration)


def get_cutoff(cell_param, element_pool, cryst_structure):
    if not cell_param[0] == 'None':
        try:
            cuttoff = float(cell_param[0])
        except Exception as err:
            raise Exception(f'{err}')
    else:
        if cryst_structure == 'fcc':
            cell_param = FaceCenteredCubic(element_pool[0]).cell.cellpar()[:3]
            cuttoff = cell_param[0]
        elif cryst_structure == 'bcc':
            cell_param = BodyCenteredCubic(element_pool[0]).cell.cellpar()[:3]
            cuttoff = cell_param[0]
    return cuttoff, cell_param


def generate_structure(my_model, output_name, element_pool, conc, cryst_structure, cell_size, cell_param):
    since = time.time()

    pathlib.Path(output_name).mkdir(parents=True, exist_ok=True)

    cutof, cell_param = get_cutoff(cell_param, element_pool, cryst_structure)

    NN_in_shell1 = coordination_numbers[cryst_structure][0]
    NN_in_shell2 = coordination_numbers[cryst_structure][1]

    GaNn = NnGa()
    AlloyGen = AlloysGen(element_pool, conc, cryst_structure)

    structureX = AlloyGen.gen_raw_crystal(cryst_structure, cell_size, element=element_pool[0], lattice_param=cell_param)

    max_diff_elements = AlloyGen.get_max_diff_elements(
        element_pool, conc, structureX.num_sites)

    configuration = AlloyGen.gen_configuration(structureX, element_pool, cutof, my_model, device='cpu',
                                               max_diff_element=max_diff_elements,
                                               constrained=True)

    # generated_structure = AseAtomsAdaptor.get_structure(configuration)

    write_to_cif(output_name, configuration)
    write_to_png(output_name, configuration)

    print('\n', '*' * 20, 'Best structure', '*' * 20)

    print('chemical formula: ', configuration.get_chemical_formula(), '\n')

    CN1_list, CN2_list = AlloyGen.get_coordination_numbers(configuration, cutof)

    # print('CN: ', CN1_list, '\n')

    shell1_fitness_AA = GaNn.get_shell1_fitness_AA(CN1_list)
    shell2_fitness_AA = GaNn.get_shell2_fitness_AA(CN2_list, NN_in_shell2)
    shell1_fitness_AB = GaNn.get_shell1_fitness_AB(CN1_list, conc, NN_in_shell1)
    max_diff_element_fitness = GaNn.get_max_diff_element_fitness(
        max_diff_elements, configuration)
    print('shell1_fitness_AA: ', sum(shell1_fitness_AA.values()), '\n')
    print('shell2_fitness_AA: ', sum(shell2_fitness_AA.values()), '\n')
    print('shell1_fitness_AB: ', sum(shell1_fitness_AB.values()), '\n')
    print('max_diff_element_fitness: ', sum(
        max_diff_element_fitness.values()), '\n')
    print('*' * 60)

    logger.info('Time for Generation:  {}'.format(time.time() - since))

    ###########################################################################

    # img1 = Image.open(os.path.join(output_name, 'structure.png'))
    # plt.clf()
    # plt.imshow(img1)


if __name__ == "__main__":

    # ========================== Read Parameters  ============================

    try:
        with open(os.path.join('./', "parameters.yml"), "r") as fr:
            parameters = yaml.safe_load(fr)
            logger.info('Parameters have been retrieved')
    except Exception as err:
        logger.error(f'{err}')
        raise Exception(f'{err}')

    nb_generation = parameters['nb_generation']
    crystal_structure = parameters['crystalstructure']
    cell_parameters = parameters['cell_parameters']
    elements_pool = parameters['elements_pool']
    concentrations = parameters['concentrations']
    rate = parameters['rate']
    alpha = parameters['alpha']
    device = parameters['device']

    nb_structures = parameters['nb_structures']
    generation = parameters['generation']
    size = parameters['size']
    # ==========================  Analyse Parameters  ============================

    # cutoff, cell_parameters = get_cutoff(cell_parameters, elements_pool, crystal_structure)

    model_path = parameters['model_path']

    # nb_atoms = np.prod(tuple(size))

    nb_species = len(elements_pool)

    # ==========================  Load the model  ============================

    # output = os.path.join(str(nb_species) + '_elemetns', 'model_' + str(nb_generation))
    output = os.path.dirname(model_path)

    model = read_model(model_path, elements_pool, device)
    for i in range(nb_structures):
        generate_structure(model, output, elements_pool, concentrations, crystal_structure, size, cell_parameters)
        shutil.move(os.path.join(output, 'structure.cif'), os.path.join(output, 'structure_{:03d}.cif'.format(i)))
        shutil.move(os.path.join(output, 'structure.png'), os.path.join(output, 'structure_{:03d}.png'.format(i)))
