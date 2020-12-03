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


from PIL import Image

# @profile
def train_policy(feedfoward, element_pool, concentrations,
                nb_generation,min_mean_fitness= 0.01 ):




    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #nn_ga = NnGa(n_structures=10, rate=0.25, alpha=0.1, device= device)

    combinations = alloy_gen.get_combination(element_pool)

    input_vectors, alloy_atoms,lattice_param = compute_input_vectors(element_pool, concentrations,cell_type,cell_size )




    max_diff_elements = alloy_gen.get_max_diff_elements(element_pool, concentrations, len(alloy_atoms))



    early_stop = False
    n_generations_stop = 6
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
    #model.eval()
    # with torch.set_grad_enabled(False):

    for generation in range(nb_generation + 1):
        # print(f'generation {n} ')
        since = time.time()

        if best_outpout is None:  # first iteration
            # create policies
            outputs = [] # list of output vectors
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

            policies_fitness = nn_ga.get_population_fitness(alloy_atoms,
                                                            concentrations,
                                                            outputs,
                                                            element_pool,
                                                            cell_type,
                                                            lattice_param,
                                                            combinations)

            # Rank the structures

            best_outpout = nn_ga.sort_population_by_fitness( outputs, policies_fitness)
            best_policies = nn_ga.sort_population_by_fitness(policies, policies_fitness)

            best_fitness = np.mean(np.array(policies_fitness))

        else:
            input_vectors, alloy_atoms,lattice_param = compute_input_vectors(element_pool, concentrations,cell_type,cell_size )


            next_generation = nn_ga.make_next_generation(best_policies)
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

            policies_fitness = nn_ga.get_population_fitness(alloy_atoms,
                                                            concentrations,
                                                            outputs,
                                                            element_pool,
                                                            cell_type,
                                                            lattice_param,
                                                            combinations)


            best_outpout = nn_ga.sort_population_by_fitness( outputs,
                                                       policies_fitness)
            best_policies = nn_ga.sort_population_by_fitness(next_generation,
                                                      policies_fitness)

            # if np.mean(np.array(policies_fitness)) <  best_fitness:
            #     best_fitness = np.mean(np.array(policies_fitness))
            #     best_gerenation = generation
            #     best_weight = best_policies

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
            #weights = best_policies[0]
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



    weights = best_policies[0]
    model.l1.weight = torch.nn.parameter.Parameter(weights)
    PATH = './model_{}.pth'.format(nb_generation)
    torch.save(model.state_dict(), PATH)


    max_fitness = np.array(max_fitness)
    mean_fitness = np.array(mean_fitness)




    #pickle.dump(max_fitness, open('max_objective_{}.pkl'.format(nb_generation), 'wb'))
    #pickle.dump(mean_fitness, open('mean_objective_{}.pkl'.format(nb_generation), 'wb'))

    # plotting
    # fig = plt.figure(figsize = (8, 8))


    plt.plot(iters, np.log(max_fitness), label='Log|(max_fitness)', linewidth=4)
    plt.plot(iters, np.log(mean_fitness), label='Log (mean_fitness)', linewidth=4)

    plt.xlabel("Generation", fontsize=15)
    plt.ylabel('')
    plt.title("Training Curve ")

    plt.legend(fontsize=12, loc='upper left')
    #plt.savefig('traces.png', dpi=600, Transparent=True)
    plt.savefig("training_{}.svg".format(nb_generation), dpi=600, transparent=True)
    plt.show()
    plt.clf()
    pickle.dump(best_policies, open('best_policies_{}.pkl'.format(nb_generation), 'wb'))
    pickle.dump(best_outpout, open('best_outpout_{}.pkl'.format(nb_generation), 'wb'))
    return best_outpout, best_policies,

def compute_input_vectors(element_pool, concentrations,cell_type,cell_size ):
    """
    """

    alloy_atoms, lattice_param = alloy_gen.gen_alloy_supercell(element_pool=element_pool,
                                                              concentrations=concentrations, types=cell_type,
                                                              size=cell_size)
    atom_list = ['Ag', 'Pd', 'Pd', 'Ag','Pd', 'Ag', 'Ag','Pd']

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
    alloy_atoms.set_chemical_symbols(atom_list)

    max_diff_elements = alloy_gen.get_max_diff_elements(element_pool, concentrations, len(alloy_atoms))

    alloy_structure = AseAtomsAdaptor.get_structure(alloy_atoms)  # Pymatgen Structure
    alloy_composition = pmg.Composition(alloy_atoms.get_chemical_formula())  # Pymatgen Composition

    # generate neighbors list
    all_neighbors_list = alloy_gen.sites_neighbor_list(alloy_structure, lattice_param)



    # get properties of each atoms
    properties = alloy_gen.get_atom_properties(alloy_structure)

    # generate input vectors
    input_vectors = alloy_gen.get_input_vectors(all_neighbors_list, properties)

    return input_vectors, alloy_atoms,lattice_param







def get_coordination_numbers(alloyAtoms,cutoff,element_pool):
    """

    """


    alloyStructure = AseAtomsAdaptor.get_structure(alloyAtoms)
    # generate neighbor list in offset [ 0, 0, 0]
    all_neighbors_list = alloy_gen.sites_neighbor_list(alloyStructure, cutoff)

    neighbor_in_offset = alloy_gen.get_neighbor_in_offset_zero(alloyStructure,cutoff)

    shell1, shell2 =  alloy_gen.count_neighbors_by_shell(all_neighbors_list,alloyAtoms, element_pool)
    offset0  = alloy_gen.count_neighbors_in_offset(neighbor_in_offset ,alloyAtoms, element_pool)


    CN1_list =alloy_gen.get_CN_list(shell1, alloyAtoms) # Coordination number
    #CN2_list = alloy_gen.get_CN_list(combinations, shell2, alloy_atoms)

    CNoffset0_list = alloy_gen.get_CN_list(offset0, alloyAtoms)

    return CN1_list, CNoffset0_list



if __name__ == "__main__":
    element_pool = ['Ag', 'Pd']
    concentrations = {'Ag': 0.5, 'Pd': 0.5}
    target_concentration = {'Ag': 0.5, 'Pd': 0.5}
    cell_type = 'fcc'
    cell_size = [2,2,2]
    combinations = []
    for i in range(len(element_pool)):
        for j in range(len(element_pool)):
            combinations.append(element_pool[i]+'-'+element_pool[j])




    # Initialize  alloy_gen
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    alloy_gen = AlloysGen(element_pool, concentrations, cell_type)



    nn_ga = NnGa(n_structures=10, rate=0.25, alpha=0.1, device= device)






    nb_generation =10


    # Feedforward
    best_outpout, best_policies = train_policy(Feedforward,
                                               element_pool, concentrations,
                                               nb_generation)


    # generate a configuration from best output




    # input_vectors, alloy_atoms, lattice_param = compute_input_vectors(element_pool,
    #                                                                   concentrations,
    #                                                                   cell_type,
    #                                                                   cell_size )


    alloy_atoms, lattice_param = alloy_gen.gen_alloy_supercell(element_pool=element_pool,
                                                              concentrations=concentrations, types=cell_type,
                                                              size=cell_size)

    max_diff_elements = alloy_gen.get_max_diff_elements(element_pool,
                                                       concentrations,
                                                       len(alloy_atoms))

    template = copy.deepcopy(alloy_atoms)


    my_struc, fractional_composition = nn_ga.gen_structure_from_output(template, best_outpout[0], element_pool,max_diff_elements)

    ###########################################################################

    print('*'*20 , 'Best structure' , '*'*20)

    print('formula: ' ,my_struc.get_chemical_formula())

    CN1_list, CNoffset0_list = get_coordination_numbers(my_struc,lattice_param,element_pool)



    print('CN: ', CN1_list, '\n')

    print('offset: ',CNoffset0_list, '\n')

    shell_fitness = nn_ga.get_shell_fitness(CN1_list,concentrations,12)
    offset_fitness = nn_ga.get_offset_fitness(CNoffset0_list)
    fitness = sum(shell_fitness.values()) + sum(offset_fitness.values())
    print('fitness: ', fitness, '\n')
    ###########################################################################



    write('random_training_structure_{}.png'.format('-'.join(map(str,cell_size))), my_struc)



    img = Image.open('random_training_structure_{}.png'.format('-'.join(map(str,cell_size))))
    plt.imshow(img)

