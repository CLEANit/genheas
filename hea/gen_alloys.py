#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Fri Oct 16 23:27:41 2020

@author: Conrard Tetsassi
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


import time
import copy
import pickle

import pymatgen as pmg
from pymatgen.io.ase import AseAtomsAdaptor

import torch
# import torch.optim as optim
# import torch.nn.init as init
from torch.autograd import Variable
# import torch.nn.functional as F

from hea.tools.feedforward import Feedforward
from hea.tools.alloysgen import AlloysGen
from hea.tools.nn_ga_model import NnGa

#import line_profiler
#profile = line_profiler.LineProfiler()




# @profile
def train_policy(feedfoward, alloy_atoms, element_list, input_vectors,
                 nb_generation):

    nn_ga = NnGa()
    input_size = np.prod(input_vectors[0].shape)  # prod(18,14)
    output_size = output_size = len(element_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.from_numpy(input_vectors).float()
    # X_tensor = torch.from_numpy(input_vectors.astype('float32'))
    input_tensor = Variable(X_tensor, requires_grad=False).to(device)

    best_policies = None  # list of output vectors
    best_weights = None

    iters = []
    max_fitness = []
    mean_fitness = []
    nb_policies = 10
    policies_fitness = None
    t_10 = 0
    #with torch.set_grad_enabled(False):
    for n in range(nb_generation + 1):
        # print(f'generation {n} ')
        since = time.time()

        if best_policies is None:  # first iteration
            # create policies
            policies = []  # list of output vectors
            policies_weights = []

            for i in range(nb_policies):
                model = feedfoward(input_size, output_size)
                model.to(device)
                model.eval()
                output_tensor = model(input_tensor)
                output_vector = output_tensor.cpu().detach().numpy()

                init_weight = copy.deepcopy(model.l1.weight.data)
                init_weight = init_weight.cpu().detach().numpy()

                policies.append(output_vector)
                policies_weights.append(init_weight)

                # get the objective functions
            policies_fitness = nn_ga.get_population_fitness(
                alloy_atoms, policies, element_list, target_concentration)

            # Rank the structures

            best_policies = nn_ga.sort_population_by_fitness(policies, policies_fitness)
            best_weights = nn_ga.sort_population_by_fitness(policies_weights, policies_fitness)

        else:

            next_generation = nn_ga.make_next_generation(best_weights, policies_fitness, rate=0.25)

            for i in range(nb_policies):
                weights = next_generation[i]
                weights_tensor = torch.from_numpy(weights).float()

                # model = feedfoward(input_size, output_size)
                model.l1.weight = torch.nn.parameter.Parameter(weights_tensor)
                model.eval()
                # model.to(device)

                output_tensor = model(input_tensor)
                output_vector = output_tensor.cpu().detach().numpy()

                init_weight = copy.deepcopy(model.l1.weight.data)
                init_weight = init_weight.cpu().detach().numpy()

                policies.append(output_vector)
                policies_weights.append(init_weight)

            policies_fitness = nn_ga.get_population_fitness(
                alloy_atoms, policies, element_list, target_concentration)

            best_policies = nn_ga.sort_population_by_fitness(policies,
                                                       policies_fitness)
            best_weights = nn_ga.sort_population_by_fitness(policies_weights,
                                                      policies_fitness)

        # save the current training information
        iters.append(n)
        max_fitness.append(np.max(np.array(policies_fitness)))
        mean_fitness.append(np.mean(np.array(policies_fitness)))  # compute *average* objective
        t_10 += time.time() - since

        # Print mean fitness and time every 10 iterations.
        if n % 10 == 0:
            print('generation {:d} *** mean fitness {:.4f} *** Time: {:.1f}s'.format(
                n, mean_fitness[n], t_10))
            t_10 = 0
            # print('*' * 20)

    # torch.save(model.state_dict(), './model_{}.pth'.format(nb_generation))
    # pickle.dump(max_fitness, open('max_objective_{}.pkl'.format(nb_generation), 'wb'))
    # pickle.dump(mean_fitness, open('mean_objective_{}.pkl'.format(nb_generation), 'wb'))

    # plotting
    # fig = plt.figure(figsize = (8, 8))

    plt.plot(iters, mean_fitness, label='mean_fitness', linewidth=4)
    plt.plot(iters, max_fitness, label='max_fitness', linewidth=4)

    plt.xlabel("Generation", fontsize=15)
    plt.ylabel('')
    plt.title("Training Curve ")

    plt.legend(fontsize=12, loc='upper left')
    # plt.savefig('traces.png', dpi=600,Transparent=True)
    #plt.savefig("training_{}.svg".format(nb_generation), dpi=600, transparent=True)
    # plt.show()

    return best_policies, best_weights


if __name__ == "__main__":
    element_list = ['Ag', 'Pd']
    concentrations = {'Ag': 0.5, 'Pd': 0.5}
    target_concentration = {'Ag': 0.5, 'Pd': 0.5}
    cell_type = 'fcc'
    cell_size = [4, 4, 4]

    # generate alloys supercell
    AlloyGen = AlloysGen(element_list, concentrations, cell_type, cell_size)

    alloy_atoms, lattice_param = AlloyGen.gen_alloy_supercell(elements=element_list,
                                                              concentrations=concentrations, types=cell_type,
                                                              size=cell_size)

    alloy_structure = AseAtomsAdaptor.get_structure(alloy_atoms)  # Pymatgen Structure
    alloy_composition = pmg.Composition(alloy_atoms.get_chemical_formula())  # Pymatgen Composition

    template = copy.deepcopy(alloy_atoms)
    # generate neighbors list
    all_neighbors_list = AlloyGen.sites_neighbor_list(alloy_structure, lattice_param)

    # get properties of each atoms
    properties = AlloyGen.get_atom_properties(alloy_structure)

    # generate input vectors
    input_vectors = AlloyGen.get_input_vectors(all_neighbors_list, properties)

    # Feedforward

    nb_generation = 20

    best_policies, best_weights = train_policy(Feedforward, template, element_list, input_vectors, nb_generation)

