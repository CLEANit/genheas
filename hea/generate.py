#!/usr/bin/env python
"""
Created on Fri Oct 16 23:27:41 2020

@author: Conrard Tetsassi
"""
import datetime
import os
import pathlib
import pickle
import shutil
import time

import torch
import yaml
from ase.io import write

from pymatgen.io.ase import AseAtomsAdaptor

from hea.tools.alloysgen import (
    AlloysGen,
    coordination_numbers,
    properties_list,
)
from hea.tools.feedforward import Feedforward
from hea.tools.log import logger
from hea.tools.nn_ga_model import NnGa


# import pymatgen as pmg
# from ase.build import bulk
# from itertools import cycle
# import numpy as np
# from pymatgen.io.ase import AseAtomsAdaptor


def read_model(path, element_pool, device):
    nn_in_shell1 = coordination_numbers[crystal_structure][0]
    nn_in_shell2 = coordination_numbers[crystal_structure][1]
    n_neighbours = nn_in_shell1 + 1 + nn_in_shell2

    inputsize = n_neighbours * len(properties_list)

    outputsize = len(element_pool)

    my_model = Feedforward(inputsize, outputsize)
    my_model.to(device)

    # weights = read_policy()
    # model.l1.weight = torch.nn.parameter.Parameter(weights)

    try:
        my_model.load_state_dict(torch.load(path))
    except Exception as err:
        logger.error(f'{err}')
        raise Exception(f'{err}')
    return my_model


def read_policy(file):
    f = open(file, 'rb')
    weights = pickle.load(f)
    return weights


def write_to_cif(name, configuration):
    write(f'{name}/structure.cif', configuration)


def write_to_png(name, configuration):
    write(f'{name}/structure.png', configuration)


def generate_structure(
        my_model,
        output_name,
        element_pool,
        conc,
        cryst_structure,
        cell_size,
        cell_param,
        constraints=True,
        cubik=False,
        direction=None,
        device='cpu',
        verbose=False,
):
    logger.info('Start Structures Generation')
    since = time.time()

    pathlib.Path(output_name).mkdir(parents=True, exist_ok=True)

    # cutof, cell_param = get_cutoff(cell_param, element_pool, cryst_structure)

    nn_in_shell1 = coordination_numbers[cryst_structure][0]
    nn_in_shell2 = coordination_numbers[cryst_structure][1]

    GaNn = NnGa()
    AlloyGen = AlloysGen(element_pool, conc, cryst_structure)

    # nb_atom = prod(cell_size)
    nb_atom = AlloyGen.get_number_of_atom(cryst_structure, cell_size, cubik=cubik, direction=direction)
    nb_atom

    max_diff_elements = AlloyGen.get_max_diff_elements(element_pool, conc, nb_atom)

    logger.info(f'number of atoms : {nb_atom}')
    logger.info(f'max_diff_elements : {max_diff_elements}')
    # name = max(max_diff_elements , key=max_diff_elements.get)

    cutof = AlloyGen.get_cutoff(element_pool[0], cryst_structure, cell_param)

    structureX = AlloyGen.gen_random_structure(
        cryst_structure,
        cell_size,
        max_diff_elements,
        lattice_param=cell_param,
        name=element_pool[0],
        cubik=cubik,
        surface=direction,
    )

    structureX = AseAtomsAdaptor.get_structure(structureX)

    configuration = AlloyGen.generate_configuration(
        structureX,
        element_pool,
        cutof,
        my_model,
        device=device,
        max_diff_element=max_diff_elements,
        constrained=constraints,
        verbose=verbose,
    )

    # generated_structure = AseAtomsAdaptor.get_structure(configuration)

    write_to_cif(output_name, configuration)
    # write_to_png(output_name, configuration)

    print('\n', '*' * 20, 'Best structure', '*' * 20)

    print('chemical formula: ', configuration.get_chemical_formula(), '\n')

    CN1_list, CN2_list = AlloyGen.get_coordination_numbers(configuration, cutof)

    # print('CN: ', cn1_list, '\n')

    shell1_fitness_AA = GaNn.get_shell1_fitness_AA(CN1_list)
    shell2_fitness_AA = GaNn.get_shell2_fitness_AA(CN2_list, nn_in_shell2)
    shell1_fitness_AB = GaNn.get_shell1_fitness_AB(CN1_list, conc, nn_in_shell1)
    max_diff_element_fitness = GaNn.get_max_diff_element_fitness(max_diff_elements, configuration)
    print('shell1_fitness_AA: {:6.2f}'.format(sum(shell1_fitness_AA.values())))
    print('shell2_fitness_AA: {:6.2f}'.format(sum(shell2_fitness_AA.values())))
    print('shell1_fitness_AB: {:6.2f}'.format(sum(shell1_fitness_AB.values())))
    print('max_diff_element_fitness: {:6.2f}'.format(sum(max_diff_element_fitness.values())))
    print('*' * 60)

    logger.info(f' <.cif> files have been saved in [{output_name}]')
    logger.info(
        'Total time for the generation h:m:s ::  {}'.format(str(datetime.timedelta(seconds=time.time() - since)))
    )

    ###########################################################################

    # img1 = Image.open(os.path.join(output_name, 'structure.png'))
    # # plt.clf()
    # plt.imshow(img1)


if __name__ == '__main__':

    # ========================== Read Parameters  ============================

    input_file = 'parameters.yml'
    try:
        with open(os.path.join('./', input_file)) as fr:
            params = yaml.safe_load(fr)
    except Exception as err:
        logger.error(f'{err}')
        raise Exception(f'{err}')

    nb_generation = params['nb_generation']
    crystal_structure = params['crystalstructure']
    cell_parameters = params['cell_parameters']
    elements_pool = params['elements_pool']
    concentrations = params['concentrations']
    rate = params['rate']
    alpha = params['alpha']
    device = params['device']

    nb_structures = params['nb_structures']
    generation = params['nb_generation']
    supercell_size = params['supercell_size']
    cube = params['cubic']
    surface = params['surfaces']

    # ==========================  Analyse Parameters  ========================

    # cutoff, cell_parameters = get_cutoff(cell_parameters, elements_pool, crystal_structure)

    policy_path = params['model_path']

    # nb_atoms = np.prod(tuple(supecell_size))

    nb_species = len(elements_pool)

    NN_in_shell1 = coordination_numbers[crystal_structure][0]
    NN_in_shell2 = coordination_numbers[crystal_structure][1]
    NNeighbours = NN_in_shell1 + 1 + NN_in_shell2

    input_size = NNeighbours * len(properties_list)

    output_size = len(elements_pool)

    # ==========================  Load the model  ============================

    # output = os.path.join(str(nb_species) + '_elemetns', 'model_' + str(nb_generation))
    output = os.path.dirname(policy_path)

    best_policy = read_policy(policy_path)

    networks = [Feedforward(input_size, output_size) for i in range(len(best_policy))]
    networks = [network.to(device) for network in networks]

    for j, w in enumerate(best_policy):
        networks[j].l1.weight = torch.nn.Parameter(w)

    for i in range(nb_structures):
        generate_structure(
            networks,
            output,
            elements_pool,
            concentrations,
            crystal_structure,
            supercell_size,
            cell_parameters,
            constraints=True,
            cubik=cube,
            direction=surface,
            verbose=True,
        )

        shutil.move(os.path.join(output, 'structure.cif'), os.path.join(output, f'structure_{i:03d}.cif'))
        # shutil.move(
        #     os.path.join(
        #         output, 'structure.png'), os.path.join(
        #         output, f'structure_{i:03d}.png'))
