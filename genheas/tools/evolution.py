import copy
import random
import sys

# from ase.data import atomic_numbers
from pprint import pprint

import numpy as np
import torch

from genheas.tools.gencrystal import AlloysGen
from genheas.tools.gencrystal import coordination_numbers
from genheas.utilities.log import logger
from pymatgen.core import Composition
from pymatgen.transformations.site_transformations import (
    ReplaceSiteSpeciesTransformation,
)
from torch.autograd import Variable
from tqdm import tqdm


class NnEa(AlloysGen):
    """ """

    def __init__(self, element_pool, concentrations, crystalstructure, rate=0.25, alpha=0.1, device="cpu"):
        """
        :param concentrations:
        :param element_pool:
        :param crystalstructure:
        :param rate: float : rate of policies to keep
        :param alpha:  parameter for gene mutation
        :param device:
        """
        super().__init__(element_pool, concentrations, crystalstructure)
        self.rate = rate
        self.alpha = alpha
        self.device = device
        self.element_pool = element_pool
        self.concentrations = concentrations
        self.crystalstructure = crystalstructure
        # self.peers = self.get_peers(element_pool)
        # self.peers = super(NnEa, self).get_peers(element_pool)
        self.NN1 = float(coordination_numbers[self.crystalstructure][0])
        self.NN2 = float(coordination_numbers[self.crystalstructure][1])
        # self.max_diff_element = max_diff_element

    # @staticmethod
    # def get_peers(element_pool):
    #     """
    #     list of  unique peer of atom in the structure
    #     """
    #     peers = []
    #     for i in range(len(element_pool)):
    #         for j in range(len(element_pool)):
    #             peers.append(element_pool[i] + "-" + element_pool[j])
    #     return peers

    def get_target_shell1(self):
        """
        :return:
        """
        target_shell1 = {}
        nb_species = len(self.concentrations.keys())
        for peer in self.peers:
            atm1, atm2 = peer.split("-")
            if atm1 == atm2:
                target_shell1[peer] = np.array([0])
            else:
                # param = self.NN1 * self.concentrations[atm2]
                target = self.NN1 * self.concentrations[atm1]  # + (param / (nb_species - 1))
                target_shell1[peer] = np.array([target], dtype="f")

        self.target_shell1 = target_shell1
        logger.info("target_shell1 Initialized")
        return target_shell1

    def get_target_shell2(self):
        """
        :return:
        """

        target_shell2 = {}
        for peer in self.peers:
            atm1, atm2 = peer.split("-")
            if atm1 == atm2:
                target_shell2[peer] = np.array([self.NN2], dtype="f")
            else:
                target_shell2[peer] = np.array([0.0], dtype="f")
        logger.info("target_shell2 Initialized")
        self.target_shell2 = target_shell2
        return target_shell2

    @staticmethod
    def count_occurrence_to_dict(arr, element_pool):
        """
        count occurrence in numpy array and
        return a list dictionary
        """
        my_list = []
        if isinstance(arr, list):
            lenght = len(arr)
        elif isinstance(arr, np.ndarray):
            lenght = arr.shape[0]
        else:
            raise Exception("arr should be an array aor a list")

        for i in range(lenght):
            unique, counts = np.unique(arr[i], return_counts=True)
            my_dict = dict(zip(unique, counts))
            for elem in element_pool:
                if elem not in my_dict.keys():
                    my_dict[elem] = 0
            my_list.append(my_dict)
        return my_list

    @staticmethod
    def get_symbols_indexes(crystal):
        """
        return dictionary with indexes of each type of atom
        """
        species = crystal.species
        species = np.array(list(map(lambda x: x.name, species)))
        species_set = np.unique(species)

        symbols_indexes = {}

        for elem in species_set:
            symbols_indexes[elem] = np.where(species == elem)[0]
        return symbols_indexes

    def _a_around_b(self, a, b, shell, crystal):
        """
        shell: list of dictionary with the neighbour of each atom
        symbols_indexes : dictionary with the indexes of each element in the structure
        a: symbol of element a
        b: symbol of element b
        return list of number of atom a around b
        """
        symbols_indexes = self.get_symbols_indexes(crystal)
        return np.array([shell[i][a] for i in symbols_indexes[b]], dtype="f")

    def get_CN_list(self, shell, crystal):
        """
        combinasons list of  unique pair of atom in the structure
        symbols_indexes : dictionary with the indexes of each element in the structure
        return dictionannry with the list of neighbor of each atom by type
        {'Ag-Ag': [6, 4, 4, 6], # Ag around Ag
         'Ag-Pd': [8, 6, 6, 8], # Ag around Pd
         'Pd-Ag': [6, 8, 8, 6],
         'Pd-Pd': [4, 6, 6, 4]}
        """
        CN_list = {}

        for peer in self.peers:
            atm1 = peer.split("-")[0]
            atm2 = peer.split("-")[1]
            try:
                CN_list[peer] = self._a_around_b(atm1, atm2, shell, crystal)
            except KeyError:
                CN_list[peer] = np.array([0.0])
        return CN_list

    @staticmethod
    def get_mae_shell(pred_shell, target_shell):
        """
        :param pred_shell:
        :param target_shell:
        :return:
        """
        shell_AA = {}
        shell_AB = {}
        for key in pred_shell.keys():
            atm1, atm2 = key.split("-")
            if atm1 == atm2:
                shell_AA[key] = NnEa.mae(pred_shell[key], target_shell[key])
            else:
                shell_AB[key] = NnEa.mae(pred_shell[key], target_shell[key])
        return sum(shell_AA.values()) / float(len(shell_AA)), sum(shell_AB.values()) / float(len(shell_AB))

    @staticmethod
    def mae(prediction, target):
        """
        Computes the mean absolute error between prediction and target
        Parameters
        ----------
        prediction: np.array (N, 1)
        target: np.array (N, 1)
        """
        return np.mean(np.abs(target - prediction))

    # @staticmethod
    # def _rmsd(v, w):
    #     """
    #     Calculate Root-mean-square deviation from two sets of vectors V and W.
    #     Parameters
    #     ----------
    #     V : array
    #         (N,D) matrix, where N is points and D is dimension.
    #     W : array
    #         (N,D) matrix, where N is points and D is dimension.
    #     Returns
    #     -------
    #     rmsd : float
    #         Root-mean-square deviation between the two vectors
    #     """
    #     diff = np.abs(np.array(v) - np.array(w))
    #     N = len(v)
    #     return np.sqrt((diff * diff).sum() / N)

    @staticmethod
    def get_composition_fitness(max_diff_element, crystal):

        species = crystal.species
        species = list(map(lambda x: x.name, species))
        species = {x: species.count(x) for x in species}

        pred, target = [], []
        for key in max_diff_element.keys():
            if key in species.keys():
                pred.append(species[key])
            else:
                pred.append(0)
            target.append(max_diff_element[key])
        fitness = NnEa.mae(np.array(pred), np.array(target))

        return fitness

    def get_population_fitness(self, configurations, max_diff_element):

        pop_fitness = []

        for configuration in configurations:
            _, _, shells = super().get_sites_neighbor_list(configuration)

            shell1 = self.count_occurrence_to_dict(shells[0], self.element_pool)
            shell2 = self.count_occurrence_to_dict(shells[1], self.element_pool)

            CN1_list = self.get_CN_list(shell1, configuration)  # Coordination number
            CN2_list = self.get_CN_list(shell2, configuration)

            shell1_fitness_AA, shell1_fitness_AB = self.get_mae_shell(CN1_list, self.target_shell1)
            # shell2_fitness_AA, shell2_fitness_AB = self.get_mae_shell(CN2_list, self.target_shell2)
            composition_fitness = self.get_composition_fitness(max_diff_element, configuration)

            fitness = (
                composition_fitness
                + shell1_fitness_AA
                # + shell1_fitness_AB
                # + shell2_fitness_AA
                # + shell2_fitness_AB)
            )

            pop_fitness.append(fitness)

        return np.array(pop_fitness)

    # @profile
    @staticmethod
    def sort_population_by_fitness(population, key):
        sorted_index = np.argsort(key)
        sorted_population = [population[idx] for idx in sorted_index]
        return sorted_population

    # @profile
    @staticmethod
    def choice_random(population, n):
        random_weights = random.choices(population, k=n)
        return random_weights

    # @profile
    def mutate(self, individual, alpha=None):
        if alpha is None:
            alpha = self.alpha
        noise = torch.FloatTensor(individual.shape).uniform_(-1, 1) * alpha
        noise = noise.to(self.device)
        noised_individual = individual.add(noise)
        return noised_individual

    # @profile
    def make_next_generation(self, previous_population, rate=None, key=None):
        """
        population is supposed to be sorted by fitness
        previous_population: list of list
        """
        # logger.info('Start Mutation')
        if rate is None:
            rate = self.rate

        if key is not None:
            sorted_by_fitness_population = self.sort_population_by_fitness(previous_population, key)
        else:
            sorted_by_fitness_population = previous_population

        population_size = len(previous_population)

        top_size = int(np.ceil(population_size * rate))

        # take top 25%
        # list of list
        top_population = sorted_by_fitness_population[:top_size]

        next_generation = top_population

        # randomly mutate the weights of the top 25% of structures to make up
        # the remaining 75%
        selections = self.choice_random(top_population, (population_size - top_size))  # list of list

        mutants = [[self.mutate(selec) for selec in selection] for selection in selections]

        # # crossover 2 random items
        # choice = random.sample(range(top_size), k=2)
        # selections[choice[0]], selections[choice[1]] = self.cxOnePoint(selections[choice[0]], selections[choice[1]])
        next_generation.extend(mutants)

        # logger.info('Mutation completed')
        return next_generation
