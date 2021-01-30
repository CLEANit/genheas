#!/usr/bin/env python
"""
Created on Fri Oct 16 23:27:41 2020

@author: Conrard TETSASSI
"""
import datetime
import glob
import multiprocessing as mp
import os
import pathlib
import pickle
import random
import time
from math import prod

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
# import torch.nn.init as init
from pymatgen.io.ase import AseAtomsAdaptor

from hea.tools.alloysgen import (
    AlloysGen,
    coordination_numbers,
    properties_list,
)
from hea.tools.feedforward import Feedforward
from hea.tools.log import logger
from hea.tools.nn_ga_model import NnGa

# Scheduler import
# from torch.optim.lr_scheduler import StepLR

# import EarlyStopping
# from pytorchtools import EarlyStopping

# Set seed
# torch.manual_seed(0)

sns.set()

opt_parameters_list = [
    'nb_atom',
    'device',
    'rate',
    'alpha',
    'nb_generation',
    'nb_policies',
    'nb_network_per_policy',
    'opt_time',
    'input_size',
    'output_size',
    'step_time',
]


# data_filepath = "training_data/{}a-{}p-{}n-{}m.csv".format(alpha,nb_policies,nb_network_per_policy)

# By default we skip the first row, which contains the headers
# By skipping 2 rows, you can disregard the first data-point (0,0) to get
# a closer look
def plot_optimization(data_filepath, skiprows=1):
    plt.close()
    for filename in glob.glob(data_filepath + '/*.csv'):
        x, y = np.loadtxt(filename, delimiter=',', unpack=True, usecols=(0, 1), skiprows=skiprows)
        plt.plot(x, y, label=os.path.basename(filename).split('.csv')[0], linewidth=3)

    plt.xlabel('Generation', fontsize=16)
    plt.ylabel('Fitness function', fontsize=16)
    plt.title('Optimization Curve', fontsize=20)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.legend(fontsize=15, loc='upper right')
    plt.savefig(f'{data_filepath}/optimization_curve.png', dpi=600, transparent=True, bbox_inches='tight')
    plt.show()


def training_0(AlloyGen, structure, element_pool, nb_network_per_policy, input_size, output_size, cutoff, device='cpu'):
    """
    :param output_size:
    :param input_size:
    :param nb_network_per_policy:
    :param AlloyGen:
    :param structure:
    :param element_pool:
    :param cutoff:
    :param device:
    :return:
    """

    networks = [Feedforward(input_size, output_size) for i in range(nb_network_per_policy)]
    networks = [network.to(device) for network in networks]

    network_weights = [network.l1.weight.data for network in networks]

    # random.shuffle(atom_list)
    # atoms.set_chemical_symbols(atom_list)

    structureX = AseAtomsAdaptor.get_structure(structure)

    configuration = AlloyGen.generate_configuration(structureX, element_pool, cutoff, networks, device=device)

    return configuration, network_weights


def training_1(
    AlloyGen,
    networks,
    network_weights,
    structure,
    element_pool,
    nb_network_per_policy,
    input_size,
    output_size,
    cutoff,
    device='cpu',
):
    """
    :param networks:
    :param network_weights:
    :param output_size:
    :param input_size:
    :param nb_network_per_policy:
    :param AlloyGen:
    :param structure:
    :param element_pool:
    :param cutoff:
    :param device:
    :return:
    """
    #

    for j, w in enumerate(network_weights):
        networks[j].l1.weight = torch.nn.Parameter(w)

    # random.shuffle(atom_list)
    # atoms.set_chemical_symbols(atom_list)

    structureX = AseAtomsAdaptor.get_structure(structure)

    configuration = AlloyGen.generate_configuration(structureX, element_pool, cutoff, networks, device=device)

    return configuration


def train_policy(
    crystal_structure,
    element_pool,
    concentrations,
    nb_generation,
    cell_size,
    cell_param,
    device='cpu',
    rate=0.25,
    alpha=0.1,
    nn_per_policy=1,
    nb_policies=8,
    oxidation_states=None,
    fitness_minimum=0,
    patience=100,
    cubik=False,
    direction=None,
):
    """
    :param direction:
    :param cubik:
    :param patience: (int) nb generations to wait before early stop
    :param fitness_minimum: (float) minimum fitness to reach
    :param oxidation_states: (dict) {'Ag': 0, 'Pd': 0}
    :param nb_generation: (int) number of generation to run
    :param nb_policies: (int) number of policies = input structures
    :param nn_per_policy: (int) number of network per policy
    :param alpha: (float) scaling factor for the weight mutation
    :param rate: (float) % of policy to keep at each generation
    :param device: cpu or gpu
    :param cell_param: (array) lattice constant of the input structure [3.61, 3.61, 3.61]
    :param cell_size: (array) size of the input structure  [4,4,4]
    :param concentrations: (dict) concentration of each element {'Ag': 0.5, 'Pd': 0.5}
    :param element_pool: list of element ['Ag', 'Pd']
    :param crystal_structure: (str) 'fcc' 'bcc' 'hpc'
    :return: list with the best weight
    """
    # ==========================  Initialization  ============================

    logger.info(f'Input parameters:: alpha [{alpha}]\t rate [{rate}] \t device [{device}]')
    # early_stop = False
    n_generations_stop = patience
    generations_no_improve = 0

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    GaNn = NnGa(rate=rate, alpha=alpha, device=device)
    AlloyGen = AlloysGen(element_pool, concentrations, crystal_structure, oxidation_states)

    # combinations = AlloyGen.get_combination(element_pool)

    nb_species = len(element_pool)
    output = os.path.join(str(nb_species) + '_elements', 'model_' + str(nb_generation))

    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    nn_in_shell1 = coordination_numbers[crystal_structure][0]
    nn_in_shell2 = coordination_numbers[crystal_structure][1]
    n_neighbours = nn_in_shell1 + 1 + nn_in_shell2

    input_size = n_neighbours * len(properties_list)

    output_size = len(element_pool)

    sorted_weights_list = None
    # iter = 0
    min_fitness = []
    nb_network_per_policy = nn_per_policy
    nb_policies = nb_policies

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # my_model = Feedforward(input_size, output_size)
    # my_model.to(device)
    since0 = time.time()
    nb_atom = AlloyGen.get_number_of_atom(crystal_structure, cell_size, cubik=cubik, direction=direction)

    max_diff_elements = AlloyGen.get_max_diff_elements(element_pool, concentrations, nb_atom)

    if not prod(cell_size) == sum(max_diff_elements.values()):
        logger.error(
            'the size : [{}] and the max_diff_elem : [{}] are not consistent'.format(
                prod(cell_size), sum(max_diff_elements.values())
            )
        )
        raise Exception(
            'the size : [{}] and the max_diff_elem : [{}] are not consistent'.format(
                nb_atom, sum(max_diff_elements.values())
            )
        )

    # atoms = AlloyGen.gen_random_structure(crystal_structure, cell_size, max_diff_elements, cell_param)
    logger.info('Generating the Input structure')
    n_input = nb_policies

    logger.info(f'max_diff_elements: {max_diff_elements}')
    logger.info(f'Number of policies: {n_input}')
    logger.info(f'Number of ANN per policy: {nn_per_policy}')
    logger.info(f'Number of atoms in the  input structure: {nb_atom}')

    cutoff = cell_param[0]
    logger.info('Start Optimization')
    # with torch.set_grad_enabled(False):
    networks = None

    for generation in range(nb_generation):
        since = time.time()
        # input_structures, _ = AlloyGen.gen_alloy_supercell(
        #     element_pool, concentrations, crystal_structure, cell_size, n_input, lattice_param=cell_param
        # )

        input_structures = [
            AlloyGen.gen_random_structure(
                crystal_structure,
                cell_size,
                max_diff_elements,
                lattice_param=cell_param,
                name=element_pool[0],
                cubik=cubik,
                surface=direction,
            )
            for _ in range(n_input)
        ]

        if networks is None:  # first iteration

            network_weights_list = []
            configurations = []

            for _, structure in enumerate(input_structures):
                networks = [Feedforward(input_size, output_size) for i in range(nb_network_per_policy)]
                networks = [network.to(device) for network in networks]

                network_weights = [network.l1.weight.data for network in networks]

                # random.shuffle(atom_list)
                # atoms.set_chemical_symbols(atom_list)

                structureX = AseAtomsAdaptor.get_structure(structure)

                configuration = AlloyGen.generate_configuration(
                    structureX, element_pool, cutoff, networks, device=device
                )

                configurations.append(configuration)
                network_weights_list.append(network_weights)

            # pool = mp.Pool(mp.cpu_count())
            # results = pool.starmap(training_0, [
            #     (AlloyGen, structure, element_pool, nb_network_per_policy, input_size, output_size, cutoff) for
            #     structure in input_structures])
            # pool.close()
            # configurations = [result[0] for result in results]
            # network_weights_list = [result[1] for result in results]

            configurations_fitness = GaNn.get_population_fitness(
                configurations, concentrations, max_diff_elements, element_pool, crystal_structure, cutoff
            )

            # Rank the policies

            sorted_weights_list = GaNn.sort_population_by_fitness(
                network_weights_list, configurations_fitness
            )  # list of list

            # sorted_configurations_fitness = GaNn.sort_population_by_fitness(configurations_fitness,
            #                                                                 configurations_fitness)
            # print(sorted_configurations_fitness)

        else:

            network_weights_list = GaNn.make_next_generation(sorted_weights_list, rate=rate)  # list of list

            configurations = []
            for i, structure in enumerate(input_structures):

                network_weights = network_weights_list[i]

                for j, w in enumerate(network_weights):
                    networks[j].l1.weight = torch.nn.Parameter(w)
                # random.shuffle(atom_list)
                # atoms.set_chemical_symbols(atom_list)

                structureX = AseAtomsAdaptor.get_structure(structure)

                configuration = AlloyGen.generate_configuration(
                    structureX, element_pool, cutoff, networks, device=device
                )

                configurations.append(configuration)

            # pool = mp.Pool(mp.cpu_count())
            # configurations = pool.starmap(training_1, [(
            #     AlloyGen, networks, network_weights, structure, element_pool, nb_network_per_policy, input_size,
            #     output_size, cutoff) for
            #     network_weights, structure in zip(network_weights_list, input_structures)])
            # pool.close()

            configurations_fitness = GaNn.get_population_fitness(
                configurations, concentrations, max_diff_elements, element_pool, crystal_structure, cutoff
            )

            # Rank the policies

            sorted_weights_list = GaNn.sort_population_by_fitness(
                network_weights_list, configurations_fitness
            )  # list of list

            # sorted_configurations_fitness = GaNn.sort_population_by_fitness(configurations_fitness,
            #                                                                 configurations_fitness)
            # print(sorted_configurations_fitness)

        # save the current training information

        step_time = time.time() - since
        min_fitness.append([generation + 1, np.min(configurations_fitness), step_time])

        # compute *average* objective
        # mean_fitness.append(
        #     [generation + 1, np.mean(configurations_fitness), step_time])

        logger.info(
            'generation: {:6d}/{} |fitness {:10.6f} *** time/step: {:.1f}s'.format(
                generation + 1, nb_generation, np.min(configurations_fitness), step_time
            )
        )

        if min_fitness[generation][1] < fitness_minimum:

            generations_no_improve = 0
            min_fitness = min_fitness[generation][1]
        # torch.save(my_model.state_dict(), PATH)
        else:
            generations_no_improve += 1

        if generation > 5 and generations_no_improve == n_generations_stop:
            logger.warning('Early stopping!')
            # early_stop = True
            # PATH = f'{output}/early_model_{nb_generation}.pth'
            logger.info('Weights Saved')
            pickle.dump(
                sorted_weights_list[0],
                open(f'{output}/ES_weights_{alpha}a-{nb_policies}p-{nb_network_per_policy}n.pkl', 'wb'),
            )
            break
        else:
            continue
        break

    logger.info('Optimization completed')
    best_policy = sorted_weights_list[0]

    pickle.dump(best_policy, open(f'{output}/BestWeights_{alpha}a-{nb_policies}p-{nb_network_per_policy}n.pkl', 'wb'))

    logger.info(f'Best policy saved in  [{output}/BestWeights_{alpha}a-{nb_policies}p-{nb_network_per_policy}n.pkl]')
    min_fitness = np.array(min_fitness)
    # mean_fitness = np.array(mean_fitness)
    opt_time = time.time() - since0
    logger.info(
        'Cumulative optimization time after  {} generations [h:m:s] ::  {}'.format(
            generation + 1, str(datetime.timedelta(seconds=opt_time))
        )
    )
    np.savetxt(
        '{}/{}a-{}p-{}n.csv'.format(output, alpha, nb_policies, nb_network_per_policy),
        min_fitness,
        delimiter=',',
        header='Generation, min_fitness, time',
    )
    # np.savetxt('{}/mean_fitness.csv'.format(output), mean_fitness, delimiter=",",
    #            header='Generation, min_fitness, time')

    plot_optimization(output, skiprows=1)

    # nb_atoms = len(input_structures[0])
    opt_parameters = {}
    for param in opt_parameters_list:
        opt_parameters[param] = eval(param)

    with open(f'{output}/opt_parameters_{alpha}a-{nb_policies}p-{nb_network_per_policy}n.yml', 'w') as yaml_file:
        yaml.dump(opt_parameters, yaml_file, default_flow_style=False)

    return sorted_weights_list[0]
