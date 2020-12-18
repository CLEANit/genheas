#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Fri Oct 16 23:27:41 2020

@author: Conrard Tetsassi
"""

from PIL import Image
from tools.nn_ga_model import NnGa
from tools.alloysgen import AlloysGen
from tools.feedforward import Feedforward
from torch.autograd import Variable
import torch.nn.init as init
import torch
from    ase import io
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.transformations.site_transformations import ReplaceSiteSpeciesTransformation
import pymatgen as pmg
import pickle
import copy
import time
from ase.io import write
from ase.lattice.cubic import FaceCenteredCubic, BodyCenteredCubic
from ase.build import bulk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pathlib
import os
sns.set()
import yaml

from tools.log import logger



# @profile



# def get_coordination_numbers(alloyAtoms, cutoff, element_pool):
#     """

#     """

#     alloyStructure = AseAtomsAdaptor.get_structure(alloyAtoms)
#     # generate neighbor list in offset [ 0, 0, 0]
#     all_neighbors_list = AlloyGen.sites_neighbor_list(alloyStructure, cutoff)

#     shell1, shell2 = AlloyGen.count_neighbors_by_shell(
#         all_neighbors_list, alloyAtoms, element_pool)

#     CN1_list = AlloyGen.get_CN_list(shell1, alloyAtoms)  # Coordination number
#     CN2_list = None
#     #CN2_list = AlloyGen.get_CN_list(shell2, alloyAtoms)

#     return CN1_list, CN2_list
# # @profile


def train_policy(
        feedfoward,
        element_pool,
        concentrations,
        nb_generation,
        input_size,
        output_size,
        train_size,
        device='cpu',
        guess_weight=None, min_mean_fitness=0.01):

    early_stop = False
    n_generations_stop = 10
    generations_no_improve = 0
    # cutoff = cutoff

    nb_species = len(element_pool)
    output = str(nb_species)+'_components_systems'

    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    sorted_policies = None
    iters = []
    max_fitness = []
    mean_fitness = []
    nb_policies = 10
    nb_config = 10
    policies_fitness = None

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = feedfoward(input_size, output_size)
    model.to(device)
    # model.eval()
    # with torch.set_grad_enabled(False):

    since0 = time.time()
    with torch.set_grad_enabled(False):
        for generation in range(nb_generation):
            # print(f'generation {n} ')
            since = time.time()

            #rand_number = random.choices([11,12])[0]
            rand_number = random.randint(1, 12)
            #rand_number =11


            atoms = io.read('training_data/fcc/clusters_{}/bestsqs_{:02d}.cif'.format(train_size,rand_number))
            max_diff_elements = AlloyGen.get_max_diff_elements(element_pool, concentrations, len(atoms))

            atom_list =[]
            for elem in element_pool:
                atom_list.extend([elem]*max_diff_elements[elem])

            random.shuffle(atom_list)
            #atom_list = [elements_pool[0]]*len(atoms)

            #atom_list = ['X']*len(atoms)

            atoms.set_chemical_symbols(atom_list)

            structureX = AseAtomsAdaptor.get_structure(atoms)
            #structureX =  AlloyGen.gen_raw_crystal(crystalstructure, cell_size,cell_parameters,element=elements_pool[0])
            #structureX = AlloyGen.gen_random_crystal_3D(crystalstructure, [elements_pool[0]], Nb_elements)

            #cell_param = np.array(structureX.lattice.abc)

            cutoff = 4.0-1

            # max_diff_elements = AlloyGen.get_max_diff_elements(
            #     elements_pool, concentrations, structureX.num_sites)

            if sorted_policies is None:  # first iteration

                policies = []
                policies_fitness = []
                for i in range(nb_policies):
                    model = feedfoward(input_size, output_size)
                    #init.normal_(model.l1.weight, mean=0, std=1)

                    if i==0 and guess_weight is not None:
                        try:
                            model.l1.weight = torch.nn.parameter.Parameter(guess_weight)
                            logger.info('+' * 5, 'policy {} initilized with guess weight'.format(i))
                            #print('+' * 5, 'policy {} initilized with guess weight'.format(i))
                        except Exception as e:
                            logger.warning('+' * 5,'Warning: error in guess weight:',print(f'e'))
                            #print('+' * 5,'Warning: error in guess weight:',print(e))
                            pass
                    # model.to(device)
                    policies.append(model.l1.weight.data)
                    if i < 1:
                        pass
                    elif torch.all(torch.eq(policies[i - 1], policies[i])):
                        logger.warning(
                            '*' * 5,
                            'Warning: Initial weight are identical to previous',
                            '*' * 5)

                    for i in range(nb_config):
                        configurations = []
                        configurations.append(
                            AlloyGen.gen_configuration(
                                structureX,
                                element_pool,
                                properties,
                                cutoff,
                                model, device=device))
                            # AlloyGen.gen_constrained_configuration(
                            #     structureX,
                            #     elements_pool,
                            #     properties,
                            #     max_diff_elements,
                            #     cutoff,
                            #     model,device='cpu'))

                    configurations_fitness = GaNn.get_population_fitness(
                        configurations,
                        concentrations,
                        max_diff_elements,
                        element_pool,
                        crystalstructure,
                        cutoff,
                        combinations)


                    policies_fitness.append(configurations_fitness)
                # Rank the policies

                policies_fitness = np.array(policies_fitness)

                sorted_policies = GaNn.sort_population_by_fitness(
                    policies, policies_fitness)

                average_fitness = np.mean(np.array(policies_fitness))

                best_policies = sorted_policies[0]

            else:

                policies = GaNn.make_next_generation(sorted_policies)

                policies_fitness = []
                for i in range(nb_policies):
                    weights = policies[i]
                    model.l1.weight = torch.nn.parameter.Parameter(weights)
                    for i in range(nb_config):
                        configurations = []
                        configurations.append(
                            AlloyGen.gen_configuration(
                                structureX,
                                element_pool,
                                properties,
                                cutoff,
                                model, device=device))
                            # AlloyGen.gen_constrained_configuration(
                            #     structureX,
                            #     elements_pool,
                            #     properties,
                            #     max_diff_elements,
                            #     cutoff,
                            #     model,device='cpu'))

                    configurations_fitness = GaNn.get_population_fitness(
                        configurations,
                        concentrations,
                        max_diff_elements,
                        element_pool,
                        crystalstructure,
                        cutoff,
                        combinations)
                    policies_fitness.append(configurations_fitness)
                # Rank the structures

                policies_fitness = np.array(policies_fitness)

                sorted_policies = GaNn.sort_population_by_fitness(
                    policies, policies_fitness)

                average_fitness = np.mean(np.array(policies_fitness))

                # Rank the policies

                policies_fitness = np.array(policies_fitness)

                sorted_policies = GaNn.sort_population_by_fitness(
                    policies, policies_fitness)

            # save the current training information
            iters.append(generation)
            max_fitness.append(np.max(np.array(policies_fitness)))
            # compute *average* objective
            mean_fitness.append(np.mean(np.array(policies_fitness)))

            # t_10 += time.time() - since

            # Print mean fitness and time every 10 iterations.
            #  if n % 10 == 0:
            # print(
            #     'generation: {:6d} | cluster {:2d}| *** mean fitness {:10.6f} *** Time: {:.1f}s'.format(
            #         generation,
            #         rand_number + 1,
            #         mean_fitness[generation],
            #         time.time() - since))
            logger.info(
                'generation: {:6d} | cluster {:2d}| *** mean fitness {:10.6f} *** Time: {:.1f}s'.format(
                    generation,
                    rand_number,
                    mean_fitness[generation],
                    time.time() - since))

            # t_10 = 0
            # print('*' * 20)
            # If the validation loss is at a minimum

            if mean_fitness[generation] < min_mean_fitness:
                # Save the model
                # weights = best_policies[0]
                # model.l1.weight = torch.nn.parameter.Parameter(weights)

                epochs_no_improve = 0
                min_mean_fitness = mean_fitness[generation]

                PATH = '{}/model_{}.pth'.format(output,nb_generation)
                logger.info('model [{}/model_{}.pth] saved'.format(output,nb_generation))
                torch.save(model.state_dict(), PATH)
            else:
                generations_no_improve += 1

            if generation > 9 and generations_no_improve == n_generations_stop:
                # print('Early stopping!')
                logger.warning('Early stopping!')
                early_stop = True
                break
            else:
                continue
            break

    weights = best_policies[0]
    model.l1.weight = torch.nn.parameter.Parameter(weights)
    PATH = '{}/model_{}.pth'.format(output,nb_generation)
    logger.info('model [{}/model_{}.pth] saved'.format(output,nb_generation))
    torch.save(model.state_dict(), PATH)
    pickle.dump(
        sorted_policies,
        open(
            '{}/best_policies_{}.pkl'.format(output,nb_generation),
            'wb'))

    logger.info('Best policy saved in  [{}/best_policies_{}.pkl]'.format(output,nb_generation))
    max_fitness = np.array(max_fitness)
    mean_fitness = np.array(mean_fitness)

    logger.info('Total traning time :  {}'.format(time.time() - since0))
    # pickle.dump(max_fitness, open('max_objective_{}.pkl'.format(nb_generation), 'wb'))
    # pickle.dump(mean_fitness, open('mean_objective_{}.pkl'.format(nb_generation), 'wb'))

    # plotting
    # fig = plt.figure(figsize = (8, 8))

    plt.plot(
        iters,
        np.log(max_fitness),
        label='Log|(max_fitness)',
        linewidth=1)
    plt.plot(
        iters,
        np.log(mean_fitness),
        label='Log (mean_fitness)',
        linewidth=1)

    plt.xlabel("Generation", fontsize=15)
    plt.ylabel('')
    plt.title("Training Curve ")

    plt.legend(fontsize=12, loc='upper left')
    # plt.savefig('traces.png', dpi=600, Transparent=True)
    plt.savefig(
        "{}/training_{}.png".format(output,nb_generation),
        dpi=600,
        transparent=True)
    plt.show()
    #plt.clf()

    # pickle.dump(
    #     best_outpout,
    #     open(
    #         'best_output_{}.pkl'.format(nb_generation),
    #         'wb'))
    return sorted_policies


if __name__ == "__main__":


    coordination_numbers = {'fcc': [12, 6, 24], 'bcc': [8, 6, 12]}


     # ========================== Read Parameters  ============================
    try:
        with open(os.path.join('./', "parameters.yml"), "r") as fr:
            parameters = yaml.safe_load(fr)
        logger.info('Parameter Read')
    except Exception as e:
        logger.error(e)
        raise Exception(e)


    nb_generation = parameters['nb_generation']
    crystalstructure = parameters['crystalstructure']
    cell_parameters = parameters['cell_parameters']
    elements_pool = parameters['elements_pool']
    concentrations = parameters['concentrations']
    rate = parameters['rate']
    alpha=parameters['alpha']
    device =parameters['device']


    nb_species = len(elements_pool)
    output = str(nb_species)+'_components_systems'

    # ==========================  Initialization  ============================


    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    AlloyGen = AlloysGen(elements_pool, concentrations, crystalstructure)

    GaNn = NnGa(rate=rate, alpha=alpha, device=device)

    # guess = pickle.load(open('best_policies_5000.pkl', 'rb'))

    #training_sizes = [(4, 4, 1), (2, 2, 4), (4, 2, 2), (2, 4, 2)]
    # ==========================  Parameters  ============================

    #cell_parameters = FaceCenteredCubic(elements_pool[0]).cell.cellpar()[:3]

    #cutoff = cell_parameters[0]


    NN_in_shell1 = coordination_numbers[crystalstructure][0]
    NN_in_shell2 = coordination_numbers[crystalstructure][1]
    NNeighbours = NN_in_shell1 +1 #+ NN_in_shell2

    combinations = AlloyGen.get_combination(elements_pool)

    training_size = 8




    # ==============================  Training ===============================

    properties = AlloyGen.get_atom_properties(elements_pool)

    # np.prod(input_vector.shape)  # prod(18,14)
    prop_keys = list(properties.keys())
    input_size = NNeighbours * len(properties[prop_keys[0]])

    output_size = len(elements_pool)

    # max_diff_element = AlloyGen.get_max_diff_elements(
    #     elements_pool, concentrations, nb_atoms)

    # Feedforward
    best_policies = train_policy(Feedforward, elements_pool, concentrations,
                                 nb_generation, input_size, output_size,training_size)

    # ============================  Generalization  ==========================
    size2 = (2,2,2)
    nb_atoms = np.prod(size2)

    cell_parameters = FaceCenteredCubic(elements_pool[0]).cell.cellpar()[:3]

    cutoff = cell_parameters[0]-1

    structureX = AlloyGen.gen_raw_crystal(
        crystalstructure,
        size2,
        cell_parameters,
        element=elements_pool[0])

    max_diff_elements = AlloyGen.get_max_diff_elements(
        elements_pool, concentrations, structureX.num_sites)

    model = Feedforward(input_size, output_size)
    model.to(device)
    weights = best_policies[0]
    model.l1.weight = torch.nn.parameter.Parameter(weights)
    # generated_configuration = AlloyGen.gen_configuration(
    #     structureX,
    #     elements_pool,
    #     properties,
    #     cutoff,
    #     model,device='cpu')
    generated_configuration = AlloyGen.gen_constrained_configuration(
        structureX,
        elements_pool,
        properties,
        max_diff_elements,
        cutoff,
        model,device='cpu')

    generated_structure = AseAtomsAdaptor.get_structure(
        generated_configuration)
    generated_structure.to(filename="{}/generated_structure_{}_{}.cif".format(output,
        '-'.join(map(str, size2)), nb_generation))

    write('{}/generated_structure_{}_{}.png'.format(output,
        '-'.join(map(str, size2)), nb_generation), generated_configuration)

    print('\n', '*' * 20, 'Best structure', '*' * 20)

    print(
        'chemical formula: ',
        generated_configuration.get_chemical_formula(),
        '\n')

    CN1_list, CN2_list = AlloyGen.get_coordination_numbers(generated_configuration, cutoff)

    # print('CN: ', CN1_list, '\n')

    shell1_fitness_AA = GaNn.get_shell1_fitness_AA(CN1_list)
    shell2_fitness_AA = GaNn.get_shell2_fitness_AA(CN2_list, NN_in_shell2)
    shell1_fitness_AB = GaNn.get_shell1_fitness_AB(CN1_list, concentrations, NN_in_shell1)
    max_diff_element_fitness = GaNn.get_max_diff_element_fitness(
        max_diff_elements, generated_configuration)
    print('shell1_fitness_AA: ', sum(shell1_fitness_AA.values()), '\n')
    print('shell2_fitness_AA: ', sum(shell2_fitness_AA.values()), '\n')
    print('shell1_fitness_AB: ', sum(shell1_fitness_AB.values()), '\n')
    print('max_diff_element_fitness: ', sum(
        max_diff_element_fitness.values()), '\n')
    print('*' * 60)
    ###########################################################################

    # img1 = Image.open('trained_structure_{}.png'.format(
    #     '-'.join(map(str, size))))

    img2 = Image.open('{}/generated_structure_{}_{}.png'.format(output,
        '-'.join(map(str, size2)), nb_generation))
    # plt.imshow(img2)

    #fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    # ax1.imshow(img1)
    ax2.imshow(img2)
