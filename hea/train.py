#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Fri Oct 16 23:27:41 2020

@author: Conrard TETSASSI
"""

import glob
import os
import pathlib
import pickle
import random
import time

import ase
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.init as init
from pymatgen.io.ase import AseAtomsAdaptor

from hea.tools.alloysgen import (AlloysGen, coordination_numbers,
                                 properties_list)
from hea.tools.feedforward import Feedforward
from hea.tools.log import logger
from hea.tools.nn_ga_model import NnGa

sns.set()



def train_policy(
        crystal_structure,
        element_pool,
        concentrations,
        nb_generation,
        train_size,
        device='cpu',
        rate=0.25,
        alpha=0.1,
        oxidation_states=None,
        guess_weight=None, min_mean_fitness=0.01):
    # ==========================  Initialization  ============================
    # early_stop = False
    n_generations_stop = 6
    generations_no_improve = 0
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    GaNn = NnGa(rate=rate, alpha=alpha, device=device)
    AlloyGen = AlloysGen(element_pool, concentrations, crystal_structure, oxidation_states)

    combinations = AlloyGen.get_combination(element_pool)

    nb_species = len(element_pool)
    output = os.path.join(str(nb_species) + '_elemetns', 'model_' + str(nb_generation))

    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    NN_in_shell1 = coordination_numbers[crystal_structure][0]
    NN_in_shell2 = coordination_numbers[crystal_structure][1]
    NNeighbours = NN_in_shell1 + 1 + NN_in_shell2

    input_size = NNeighbours * len(properties_list)

    output_size = len(element_pool)

    sorted_policies = None
    iters = []
    max_fitness = []
    mean_fitness = []
    nb_policies = 10
    nb_config = 10
    # policies_fitness = None

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    my_model = Feedforward(input_size, output_size)
    my_model.to(device)

    since0 = time.time()
    with torch.set_grad_enabled(False):
        for generation in range(nb_generation):
            since = time.time()

            # rand_number = random.randint(1,10)
            name = 'structure'
            files = glob.glob('training_data/{}/clusters_{}/{}_*.cif'.format(crystal_structure, train_size, name))
            structure = random.choice(files)
            atoms = ase.io.read(structure)
            max_diff_elements = AlloyGen.get_max_diff_elements(element_pool, concentrations, len(atoms))

            atom_list = [element_pool[0]] * len(atoms)

            atoms.set_chemical_symbols(atom_list)

            structureX = AseAtomsAdaptor.get_structure(atoms)
            # structureX =  AlloyGen.gen_raw_crystal(crystalstructure, cell_size,cell_parameters,
            # element=elements_pool[0]) structureX = AlloyGen.gen_random_crystal_3D(crystalstructure, [elements_pool[
            # 0]], Nb_elements)

            # cell_param = np.array(structureX.lattice.abc)

            cutoff = 4.0  # by default train cell was created

            # max_diff_elements = AlloyGen.get_max_diff_elements(element_pool, concentrations, structureX.num_sites)

            if sorted_policies is None:  # first iteration

                policies_fitnesses = []
                policies = []
                for i in range(nb_policies):

                    my_model = Feedforward(input_size, output_size)
                    # init.normal_(model.l1.weight, mean=0, std=1)

                    if i == 0 and guess_weight is not None:
                        try:
                            my_model.l1.weight = torch.nn.parameter.Parameter(guess_weight)
                            logger.info('+' * 5, 'policy {} initilized with guess weight'.format(i))
                            # print('+' * 5, 'policy {} initilized with guess weight'.format(i))
                        except Exception as err:
                            logger.warning('Warning: error in guess weight')
                            logger.error(f'{err}')
                            # print('+' * 5,'Warning: error in guess weight:',print(e))
                            pass
                    # model.to(device)
                    policies.append(my_model.l1.weight.data)
                    if i < 1:
                        pass
                    elif torch.all(torch.eq(policies[i - 1], policies[i])):
                        logger.warning(
                            '*' * 5,
                            'Warning: Initial weight are identical to previous',
                            '*' * 5)
                    configurations = []
                    for _ in range(nb_config):
                        configurations.append(
                            AlloyGen.gen_configuration(structureX, element_pool, cutoff, my_model, device=device))

                    configurations_fitness = GaNn.get_population_fitness(
                        configurations,
                        concentrations,
                        max_diff_elements,
                        element_pool,
                        crystal_structure,
                        cutoff)

                    policies_fitnesses.append(configurations_fitness)  # list of array

                policies_avg_fitnesses = [np.mean(array) for array in policies_fitnesses]

                policies_avg_fitnesses = np.array(policies_avg_fitnesses)

                # Rank the policies

                sorted_policies = GaNn.sort_population_by_fitness(
                    policies, policies_avg_fitnesses)

                sorted_policies_fitnesses = GaNn.sort_population_by_fitness(
                    policies_fitnesses, policies_avg_fitnesses)
                # best_policies = sorted_policies[0]

            else:

                policies = GaNn.make_next_generation(sorted_policies)

                policies_fitnesses = []
                for _ in range(nb_policies):
                    weights = policies[i]
                    my_model.l1.weight = torch.nn.parameter.Parameter(weights)

                    configurations = []
                    for _ in range(nb_config):
                        configurations.append(
                            AlloyGen.gen_configuration(
                                structureX,
                                element_pool,
                                cutoff,
                                my_model, device=device))

                    configurations_fitness = GaNn.get_population_fitness(
                        configurations,
                        concentrations,
                        max_diff_elements,
                        element_pool,
                        crystal_structure,
                        cutoff)

                    policies_fitnesses.append(configurations_fitness)  # list of array

                policies_avg_fitnesses = [np.mean(array) for array in policies_fitnesses]

                policies_avg_fitnesses = np.array(policies_avg_fitnesses)

                # Rank the policies

                sorted_policies = GaNn.sort_population_by_fitness(
                    policies, policies_avg_fitnesses)

                sorted_policies_fitnesses = GaNn.sort_population_by_fitness(
                    policies_fitnesses, policies_avg_fitnesses)

            # save the current training information

            iters.append(generation)
            max_fitness.append(np.max(sorted_policies_fitnesses[0]))

            # compute *average* objective
            mean_fitness.append(np.mean(sorted_policies_fitnesses[0]))

            logger.info(
                'generation: {:6d} | *** mean fitness {:10.6f} *** Time: {:.1f}s'.format(
                    generation,
                    mean_fitness[generation],
                    time.time() - since))

            if mean_fitness[generation] < min_mean_fitness:

                generations_no_improve = 0
                min_mean_fitness = mean_fitness[generation]

                PATH = '{}/early_model_{}.pth'.format(output, nb_generation)
                logger.info('model [{}/early_model_{}.pth] saved'.format(output, nb_generation))
                torch.save(my_model.state_dict(), PATH)
            else:
                generations_no_improve += 1

            if generation > 5 and generations_no_improve == n_generations_stop:
                logger.warning('Early stopping!')
                # early_stop = True
                break
            else:
                continue
            break

    weights = sorted_policies[0]
    my_model.l1.weight = torch.nn.parameter.Parameter(weights)
    PATH = '{}/model_{}.pth'.format(output, nb_generation)
    logger.info('model [{}/model_{}.pth] saved'.format(output, nb_generation))
    torch.save(my_model.state_dict(), PATH)
    pickle.dump(
        weights,
        open(
            '{}/best_policy_{}.pkl'.format(output, nb_generation),
            'wb'))

    logger.info('Best policy saved in  [{}/best_policy_{}.pkl]'.format(output, nb_generation))
    max_fitness = np.array(max_fitness)
    mean_fitness = np.array(mean_fitness)

    logger.info('Total traning time :  {}'.format(time.time() - since0))
    # pickle.dump(max_fitness, open('max_objective_{}.pkl'.format(nb_generation), 'wb'))
    # pickle.dump(mean_fitness, open('mean_objective_{}.pkl'.format(nb_generation), 'wb'))

    # plotting
    # fig = plt.figure(figsize = (8, 8))

    plt.clf()
    plt.plot(
        iters,
        np.log(max_fitness),
        label='Log (max_fitness)',
        linewidth=1)
    plt.plot(
        iters,
        np.log(mean_fitness),
        label='Log (mean_fitness)',
        linewidth=1)

    plt.xlabel("Generation", fontsize=15)
    plt.ylabel('')
    plt.title("Fitness of the best policy ")

    plt.legend(fontsize=12, loc='upper left')
    plt.savefig(
        "{}/training_{}.png".format(output, nb_generation),
        dpi=600,
        transparent=True, bbox_inches='tight')

    plt.show()

    return sorted_policies[0]
