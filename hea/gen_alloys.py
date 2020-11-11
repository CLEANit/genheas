#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Fri Oct 16 23:27:41 2020

@author: Conrard Tetsassi
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from ase.io import write

import time
import copy
import pickle

import pymatgen as pmg
from pymatgen.io.ase import AseAtomsAdaptor

import torch
# import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
# import torch.nn.functional as F

from hea.tools.feedforward import Feedforward
from hea.tools.alloysgen import AlloysGen
from hea.tools.nn_ga_model import NnGa

#import line_profiler
#profile = line_profiler.LineProfiler()




# @profile
def train_policy(feedfoward,alloy_atoms, element_list, concentrations, input_vectors,
                 nb_generation,min_mean_fitness= 0.01 ):


    early_stop = False
    n_generations_stop = 6
    generations_no_improve = 0


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

    model = feedfoward(input_size, output_size)
    model.to(device)
    #model.eval()
    # with torch.set_grad_enabled(False):
    nn_ga = NnGa( AlloyGen,n_structures=10, rate=0.25, alpha=0.1, device= device)
    for generation in range(nb_generation + 1):
        # print(f'generation {n} ')
        since = time.time()

        if best_policies is None:  # first iteration
            # create policies
            policies = []  # list of output vectors
            policies_weights = []

            for i in range(nb_policies):
                with torch.set_grad_enabled(False):
                    init.normal_(model.l1.weight, mean=0, std=1)
                    output_tensor = model(input_tensor)
                    # output_vector = output_tensor.cpu().detach().numpy()

                    # init_weight = copy.deepcopy(model.l1.weight.data)
                    # init_weight = init_weight.cpu().detach().numpy()
                    # init_weight = model.l1.weight.data

                    policies.append(output_tensor)
                    policies_weights.append(model.l1.weight.data)

                # get the objective functions
            policies_fitness = nn_ga.get_population_fitness(alloy_atoms, policies, element_list, concentrations,cell_type,lattice_param,combinasons)

            # Rank the structures

            best_policies = nn_ga.sort_population_by_fitness(policies, policies_fitness)
            best_weights = nn_ga.sort_population_by_fitness(policies_weights, policies_fitness)

            best_fitness = np.mean(np.array(policies_fitness))

        else:

            next_generation = nn_ga.make_next_generation(best_weights)
            policies = []
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

                    policies.append(output_tensor)
                    # policies_weights.append(init_weight)

            policies_fitness = nn_ga.get_population_fitness(alloy_atoms, policies, element_list, concentrations,cell_type,lattice_param,combinasons)

            best_policies = nn_ga.sort_population_by_fitness(policies,
                                                       policies_fitness)
            best_weights = nn_ga.sort_population_by_fitness(next_generation,
                                                      policies_fitness)

            if np.mean(np.array(policies_fitness)) <  best_fitness:
                best_fitness = np.mean(np.array(policies_fitness))
                best_gerenation = generation
                best_weight = best_weights

        # save the current training information
        iters.append(generation)
        max_fitness.append(np.max(np.array(policies_fitness)))
        mean_fitness.append(np.mean(np.array(policies_fitness)))  # compute *average* objective


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
            #weights = best_weights[0]
            #model.l1.weight = torch.nn.parameter.Parameter(weights)

            epochs_no_improve = 0
            min_mean_fitness = mean_fitness[generation]


            PATH = './model_{}.pth'.format(nb_generation)
            torch.save(model.state_dict(), PATH)
        else:
             generations_no_improve += 1

        if generation > 5 and generations_no_improve == n_generations_stop:
            print('Early stopping!' )
            early_stop = True
            break
        else:
            continue
        break



    weights = best_weights[0]
    model.l1.weight = torch.nn.parameter.Parameter(weights)
    PATH = './model_{}.pth'.format(nb_generation)
    torch.save(model.state_dict(), PATH)


    max_fitness = np.array(max_fitness)
    mean_fitness = np.array(mean_fitness)




    #pickle.dump(max_fitness, open('max_objective_{}.pkl'.format(nb_generation), 'wb'))
    #pickle.dump(mean_fitness, open('mean_objective_{}.pkl'.format(nb_generation), 'wb'))

    # plotting
    # fig = plt.figure(figsize = (8, 8))

    plt.plot(iters, np.log(mean_fitness), label='Log(mean_fitness)', linewidth=4)
    plt.plot(iters, np.log(max_fitness), label='Log(max_fitness)', linewidth=4)

    plt.xlabel("Generation", fontsize=15)
    plt.ylabel('')
    plt.title("Training Curve ")

    plt.legend(fontsize=12, loc='upper left')
    #plt.savefig('traces.png', dpi=600, Transparent=True)
    plt.savefig("training_{}.svg".format(nb_generation), dpi=600, transparent=True)
    # plt.show()
    pickle.dump(best_weights, open('best_weights_{}.pkl'.format(nb_generation), 'wb'))
    return best_policies, best_weights,


if __name__ == "__main__":
    element_list = ['Ag', 'Pd']
    concentrations = {'Ag': 0.5, 'Pd': 0.5}
    target_concentration = {'Ag': 0.5, 'Pd': 0.5}
    cell_type = 'fcc'
    cell_size = [4, 4, 4]
    combinasons = []
    for i in range(len(element_list)):
        for j in range(len(element_list)):
            combinasons.append(element_list[i]+'-'+element_list[j])

    # generate alloys supercell
    AlloyGen = AlloysGen(element_list, concentrations, cell_type, cell_size)

    alloy_atoms, lattice_param = AlloyGen.gen_alloy_supercell(elements=element_list,
                                                              concentrations=concentrations, types=cell_type,
                                                              size=cell_size)
    # atom_list = ['Ag','Pd']*32
    # atom_list = ['Ag', 'Pd', 'Ag', 'Pd',
    #               'Pd', 'Ag', 'Pd', 'Ag',
    #               'Ag', 'Pd', 'Ag','Pd',
    #               'Pd', 'Ag', 'Pd', 'Ag',
    #               'Pd', 'Ag', 'Pd', 'Ag',
    #               'Ag', 'Pd','Ag', 'Pd',
    #               'Pd', 'Ag','Pd', 'Ag',
    #               'Ag', 'Pd', 'Ag','Pd',
    #               'Ag', 'Pd', 'Ag','Pd',
    #               'Pd', 'Ag','Pd', 'Ag',
    #               'Ag', 'Pd', 'Ag','Pd',
    #               'Pd', 'Ag','Pd', 'Ag',
    #               'Pd', 'Ag','Pd', 'Ag',
    #               'Ag', 'Pd', 'Ag','Pd',
    #               'Pd', 'Ag','Pd', 'Ag',
    #               'Ag', 'Pd', 'Ag','Pd']




    # alloy_atoms.set_chemical_symbols(atom_list)
    write('input_configuration.png', alloy_atoms)
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

    nb_generation =4000

    best_policies, best_weights = train_policy(Feedforward, template,
                                               element_list, concentrations,
                                               input_vectors, nb_generation)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_ga = NnGa( AlloyGen,n_structures=10, rate=0.25, alpha=0.1, device= device)
    my_struc, fractional_composition = nn_ga.gen_structure(template, best_policies[0], element_list)
    print('formula: ' ,my_struc.get_chemical_formula())
    write('random_training_structure.png', my_struc)

    from PIL import Image
    img = Image.open('random_training_structure.png')
    img.show()

