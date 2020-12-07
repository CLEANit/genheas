#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Fri Oct 16 23:27:41 2020

@author: Conrard Tetsassi
"""

from PIL import Image
from hea.tools.nn_ga_model import NnGa
from hea.tools.alloysgen import AlloysGen
from hea.tools.feedforward import Feedforward
from torch.autograd import Variable
import torch.nn.init as init
import torch
from pymatgen.io.ase import AseAtomsAdaptor
import pymatgen as pmg
import pickle
import copy
import time
from ase.io import write
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# import torch.optim as optim
# import torch.nn.functional as F


# import line_profiler
# profile = line_profiler.LineProfiler()


# @profile


def train_policy(feedfoward, element_pool, concentrations,
                 nb_generation, min_mean_fitness=0.01):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # GaNn = NnGa(n_structures=10, rate=0.25, alpha=0.1, device= device)

    combinations = AlloyGen.get_combination(element_pool)

    input_vectors, alloy_atoms, lattice_param = compute_input_vectors(
        element_pool, concentrations, crystalstructure, training_size, cell_parameters)

    cutoff = lattice_param[0]
    max_diff_elements = AlloyGen.get_max_diff_elements(
        element_pool, concentrations, len(alloy_atoms))

    early_stop = False
    n_generations_stop = 10
    generations_no_improve = 0

    input_size = np.prod(input_vectors[0].shape)  # prod(18,14)
    output_size = output_size = len(element_pool)

    X_tensor = torch.from_numpy(input_vectors).float()
    # X_tensor = torch.from_numpy(input_vectors.astype('float32'))
    input_tensor = Variable(X_tensor, requires_grad=False).to(device)

    best_outpout = None  # list of output vectors
    best_policies = None

    iters = []
    max_fitness = []
    mean_fitness = []
    nb_policies = 10
    policies_fitness = None
    t_10 = 0

    model = feedfoward(input_size, output_size)
    model.to(device)
    # model.eval()
    # with torch.set_grad_enabled(False):

    for generation in range(nb_generation + 1):
        # print(f'generation {n} ')
        since = time.time()

        if best_outpout is None:  # first iteration
            # create policies
            outputs = []  # list of output vectors
            policies = []

            for i in range(nb_policies):
                with torch.set_grad_enabled(False):
                    init.normal_(model.l1.weight, mean=0, std=1)
                    output_tensor = model(input_tensor)
                    # output_vector = output_tensor.cpu().detach().numpy()

                    # init_weight = copy.deepcopy(model.l1.weight.data)
                    # init_weight = init_weight.cpu().detach().numpy()
                    # init_weight = model.l1.weight.data

                    outputs.append(output_tensor)
                    policies.append(model.l1.weight.data)

                # get the objective functions

            policies_fitness = GaNn.get_population_fitness(alloy_atoms,
                                                           concentrations,
                                                           outputs,
                                                           element_pool,
                                                           crystalstructure,
                                                           cutoff,
                                                           combinations)

            # Rank the structures

            best_outpout = GaNn.sort_population_by_fitness(
                outputs, policies_fitness)
            best_policies = GaNn.sort_population_by_fitness(
                policies, policies_fitness)

            best_fitness = np.mean(np.array(policies_fitness))

        else:
            input_vectors, alloy_atoms, lattice_param = compute_input_vectors(
                element_pool, concentrations, crystalstructure, training_size, cell_parameters)

            cutoff = lattice_param[0]
            next_generation = GaNn.make_next_generation(best_policies)
            outputs = []
            for i in range(nb_policies):
                with torch.set_grad_enabled(False):
                    weights = next_generation[i]
                    # weights_tensor = torch.from_numpy(weights).float().to(device)

                    # model = feedfoward(input_size, output_size)
                    model.l1.weight = torch.nn.parameter.Parameter(weights)

                    # model.eval()
                    # model.to(device)

                    output_tensor = model(input_tensor)
                    # output_vector = output_tensor.cpu().detach().numpy()

                    # init_weight = copy.deepcopy(model.l1.weight.data)
                    # init_weight = init_weight.cpu().detach().numpy()

                    outputs.append(output_tensor)
                    # policies.append(init_weight)

            policies_fitness = GaNn.get_population_fitness(alloy_atoms,
                                                           concentrations,
                                                           outputs,
                                                           element_pool,
                                                           crystalstructure,
                                                           cutoff,
                                                           combinations)

            best_outpout = GaNn.sort_population_by_fitness(outputs,
                                                           policies_fitness)
            best_policies = GaNn.sort_population_by_fitness(next_generation,
                                                            policies_fitness)

            # if np.mean(np.array(policies_fitness)) <  best_fitness:
            #     best_fitness = np.mean(np.array(policies_fitness))
            #     best_gerenation = generation
            #     best_weight = best_policies

        # save the current training information
        iters.append(generation)
        max_fitness.append(np.max(np.array(policies_fitness)))
        # compute *average* objective
        mean_fitness.append(np.mean(np.array(policies_fitness)))

        # t_10 += time.time() - since

        # Print mean fitness and time every 10 iterations.
        #  if n % 10 == 0:
        print('generation {:4d} *** mean fitness {:.6f} *** Time: {:.1f}s'.format(
            generation, mean_fitness[generation], time.time() - since))

        # t_10 = 0
        # print('*' * 20)
        # If the validation loss is at a minimum

        if mean_fitness[generation] < min_mean_fitness:
            # Save the model
            # weights = best_policies[0]
            # model.l1.weight = torch.nn.parameter.Parameter(weights)

            epochs_no_improve = 0
            min_mean_fitness = mean_fitness[generation]

            PATH = './model_{}.pth'.format(nb_generation)
            torch.save(model.state_dict(), PATH)
        else:
            generations_no_improve += 1

        if generation > 9 and generations_no_improve == n_generations_stop:
            print('Early stopping!')
            early_stop = True
            break
        else:
            continue
        break

    weights = best_policies[0]
    model.l1.weight = torch.nn.parameter.Parameter(weights)
    PATH = './model_{}.pth'.format(nb_generation)
    torch.save(model.state_dict(), PATH)

    max_fitness = np.array(max_fitness)
    mean_fitness = np.array(mean_fitness)

    # pickle.dump(max_fitness, open('max_objective_{}.pkl'.format(nb_generation), 'wb'))
    # pickle.dump(mean_fitness, open('mean_objective_{}.pkl'.format(nb_generation), 'wb'))

    # plotting
    # fig = plt.figure(figsize = (8, 8))

    plt.plot(
        iters,
        np.log(max_fitness),
        label='Log|(max_fitness)',
        linewidth=4)
    plt.plot(
        iters,
        np.log(mean_fitness),
        label='Log (mean_fitness)',
        linewidth=4)

    plt.xlabel("Generation", fontsize=15)
    plt.ylabel('')
    plt.title("Training Curve ")

    plt.legend(fontsize=12, loc='upper left')
    # plt.savefig('traces.png', dpi=600, Transparent=True)
    plt.savefig(
        "training_{}.svg".format(nb_generation),
        dpi=600,
        transparent=True)
    plt.show()
    plt.clf()
    pickle.dump(
        best_policies,
        open(
            'best_policies_{}.pkl'.format(nb_generation),
            'wb'))
    pickle.dump(
        best_outpout,
        open(
            'best_output_{}.pkl'.format(nb_generation),
            'wb'))
    return best_outpout, best_policies,


def compute_input_vectors(
        element_pool,
        concentrations,
        crystalstructure,
        training_size,
        cell_parameters):
    """

    """

    alloy_atoms, lattice_param = AlloyGen.gen_alloy_supercell(element_pool=element_pool,
                                                              concentrations=concentrations, crystalstructure=crystalstructure,
                                                              size=training_size, lattice_param=cell_parameters)

    cutoff = lattice_param[0]

    max_diff_elements = AlloyGen.get_max_diff_elements(
        element_pool, concentrations, len(alloy_atoms))

    alloy_structure = AseAtomsAdaptor.get_structure(
        alloy_atoms)  # Pymatgen Structure
    alloy_composition = pmg.Composition(
        alloy_atoms.get_chemical_formula())  # Pymatgen Composition

    # generate neighbors list
    all_neighbors_list = AlloyGen.sites_neighbor_list(alloy_structure, cutoff)

    # get properties of each atoms
    properties = AlloyGen.get_atom_properties(alloy_structure)

    # generate input vectors
    input_vectors = AlloyGen.get_input_vectors(all_neighbors_list, properties)

    return input_vectors, alloy_atoms, lattice_param


def get_coordination_numbers(alloyAtoms, cutoff, element_pool):
    """

    """

    alloyStructure = AseAtomsAdaptor.get_structure(alloyAtoms)
    # generate neighbor list in offset [ 0, 0, 0]
    all_neighbors_list = AlloyGen.sites_neighbor_list(alloyStructure, cutoff)

    neighbor_in_offset = AlloyGen.get_neighbor_in_offset_zero(
        alloyStructure, cutoff)

    shell1, shell2 = AlloyGen.count_neighbors_by_shell(
        all_neighbors_list, alloyAtoms, element_pool)
    offset0 = AlloyGen.count_neighbors_in_offset(
        neighbor_in_offset, alloyAtoms, element_pool)

    CN1_list = AlloyGen.get_CN_list(shell1, alloyAtoms)  # Coordination number
    # CN2_list =AlloyGen.get_CN_list(combinations, shell2, alloy_atoms)

    CNoffset0_list = AlloyGen.get_CN_list(offset0, alloyAtoms)

    return CN1_list, CNoffset0_list


if __name__ == "__main__":

    # elements_pool = ['Ag', 'Pd']
    # concentrations = {'Ag': 0.5, 'Pd': 0.5}
    # cell_parameters = [None, None, None]

    elements_pool = ['Cu', 'Ni', 'Co', 'Cr']
    concentrations = {'Cu': 0.25, 'Ni': 0.25, 'Co': 0.25, 'Cr': 0.25}
    cell_parameters = [None, None, None]

    # elements_pool = ['Co', 'Cr', 'Fe', 'Mn', 'Ni']
    # concentrations = {'Co': 0.2, 'Cr': 0.2, 'Fe': 0.2, 'Mn': 0.2, 'Ni': 0.2}
    # cell_parameters = [3.54, 3.54, 3.54]

    coordination_numbers = {'fcc': [12, 6, 24], 'bcc': [8, 6, 12]}
    # target_concentration = {'Ag': 0.5, 'Pd': 0.5}
    crystalstructure = 'fcc'

    size = [8, 8]

    #########################################################################

    combinations = []
    for i in range(len(elements_pool)):
        for j in range(len(elements_pool)):
            combinations.append(elements_pool[i] + '-' + elements_pool[j])

    nb_atoms = np.prod(size)

    training_size = [len(elements_pool)] * len(size)

    a = size[0] / training_size[0]
    b = size[1] / training_size[1]
    try:
        c = size[2] / training_size[2]
    except BaseException:
        c = 1

    # Initialize AlloyGen
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    AlloyGen = AlloysGen(elements_pool, concentrations, crystalstructure)

    GaNn = NnGa(n_structures=10, rate=0.25, alpha=0.1, device=device)

    nb_generation = 1000

    # Feedforward
    best_outpout, best_policies = train_policy(Feedforward,
                                               elements_pool, concentrations,
                                               nb_generation)

    # generate a configuration from best output

    # input_vectors, alloy_atoms, lattice_param = compute_input_vectors(element_pool,
    #                                                                   concentrations,
    #                                                                   crystalstructure,
    # training_size )

    alloy_atoms, lattice_param = AlloyGen.gen_alloy_supercell(
        elements_pool, concentrations, crystalstructure, training_size, cell_parameters)

    cutoff = lattice_param[0]
    max_diff_elements = AlloyGen.get_max_diff_elements(elements_pool,
                                                       concentrations,
                                                       len(alloy_atoms))

    template = copy.deepcopy(alloy_atoms)

    my_Atoms, fractional_composition = GaNn.gen_structure_from_output(
        template, best_outpout[0], elements_pool, max_diff_elements)

    ###########################################################################

    my_struc = AseAtomsAdaptor.get_structure(my_Atoms)

    Best_structrure = my_struc * (a, b, c)

    my_struc.to(filename="trained_structure{}.cif".format(nb_generation))
    Best_structrure.to(filename="Best_structrure_{}.cif".format(nb_generation))

    print('\n', '*' * 20, 'Best structure', '*' * 20)

    print('chemical formula: ', AseAtomsAdaptor.get_atoms(
        Best_structrure).get_chemical_formula(), '\n')

    CN1_list, CNoffset0_list = get_coordination_numbers(
        AseAtomsAdaptor.get_atoms(Best_structrure), cutoff, elements_pool)

    # print('CN: ', CN1_list, '\n')

    shell_fitness = GaNn.get_shell_fitness(CN1_list)

    fitness = sum(shell_fitness.values())
    print('structure  fitness: ', fitness, '\n')
    print('*' * 60)
    ###########################################################################

    write('trained_structure_{}.png'.format(
        '-'.join(map(str, training_size))), my_Atoms)

    write('Best_structure_{}.png'.format('-'.join(map(str, size))),
          AseAtomsAdaptor.get_atoms(Best_structrure))

    img1 = Image.open('trained_structure_{}.png'.format(
        '-'.join(map(str, training_size))))

    img2 = Image.open('Best_structure_{}.png'.format('-'.join(map(str, size))))
    # plt.imshow(img2)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax1.imshow(img1)
    ax2.imshow(img2)
