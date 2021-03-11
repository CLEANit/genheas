import random

import numpy as np
import pymatgen as pmg
import torch
from hea.tools.alloysgen import AlloysGen, coordination_numbers

from hea.tools.log import logger


# from ase.data import atomic_numbers


class NnGa:
    """"""

    def __init__(self, rate=0.25, alpha=0.1, device='cpu'):
        """
        #nb_polocies: int : nber of polocies generated at each generation
        n_structures: int :  nber of structure generated for each policies
        rate: float : rate of policies to keep
        alpha: parameter for gene mutation
        """
        self.rate = rate
        self.alpha = alpha
        self.device = device
        # self.max_diff_element = max_diff_element

    @staticmethod
    def _combination(element_pool):
        """
        combinations list of  unique pair of atom in the structure
        """
        combinations = []
        for i in range(len(element_pool)):
            for j in range(len(element_pool)):
                combinations.append(element_pool[i] + '-' + element_pool[j])
        return combinations

    @staticmethod
    def _rmsd(v, w):
        """
        Calculate Root-mean-square deviation from two sets of vectors V and W.
        Parameters
        ----------
        V : array
            (N,D) matrix, where N is points and D is dimension.
        W : array
            (N,D) matrix, where N is points and D is dimension.
        Returns
        -------
        rmsd : float
            Root-mean-square deviation between the two vectors
        """
        diff = np.abs(np.array(v) - np.array(w))
        N = len(v)
        return np.sqrt((diff * diff).sum() / N)

    def get_shell1_fitness_AA(self, cn1_list):
        """
        Minimize the $N_{aa}$ in the first coordination shell
         CN_list:list of dictionanry
        """
        fitness = {}
        target = 0

        for key, val in cn1_list.items():
            atm1, atm2 = key.split('-')
            if atm1 == atm2:
                fitness[key] = self._rmsd(val, target)

        return fitness

    def get_shell2_fitness_AA(self, cn1_list, n_neighbours):
        """
        Maximize the $N_{aa}$ in the second coordination shell
        CN_list:list of dictionanry
        """
        fitness = {}
        target = n_neighbours

        for key, val in cn1_list.items():
            atm1, atm2 = key.split('-')
            if atm1 == atm2:
                fitness[key] = self._rmsd(val, target)

        return fitness

    def get_shell1_fitness_AB(self, cn1_list, concentrations, n_neighbours):
        """
        Homogenize the $N_{a b}$ in the first coordination shell

        CN_list:list of dictionanry
        alloy_atoms: structure(Atoms class)
        n_neighbours:  ( number of neighbours in the shell)
        return a dictionary:


        """

        fitness = {}
        nb_species = len(concentrations.keys())

        for key, val in cn1_list.items():
            atm1, atm2 = key.split('-')
            if atm1 != atm2:
                param = n_neighbours * concentrations[atm2]
                target = n_neighbours * concentrations[atm1] + (param / (nb_species - 1))

                fitness[key] = self._rmsd(val, target)

        return fitness

    def get_max_diff_element_fitness(self, max_diff_element, alloy_atoms):
        """
        max_diff_element : maximun number of each specie

        """
        # element_list = alloy_atoms.get_chemical_symbols()
        # max_diff_elem = {x: element_list.count(x) for x in element_list}
        fitness = {}
        comp = pmg.Composition(alloy_atoms.get_chemical_formula())
        max_diff_elem = comp.as_dict()

        for key, val in max_diff_elem.items():
            fitness[key] = self._rmsd([val], [max_diff_element[key]])

        return fitness

    def get_composition_fitness(self, max_diff_element, alloy_atoms):
        """
        max_diff_element : maximun number of each specie

        """
        target_comp = pmg.Composition(max_diff_element)

        comp = pmg.Composition(alloy_atoms.get_chemical_formula())

        fitness = {}

        comp_dict = {x.name: comp.get_atomic_fraction(x) for x in comp.elements}
        target_comp_dict = {x.name: target_comp.get_atomic_fraction(x) for x in target_comp.elements}

        for key, val in comp_dict.items():
            fitness[key] = self._rmsd([val], [target_comp_dict[key]])

        return fitness

    # @profile

    def get_population_fitness(self, configurations, concentrations, max_diff_element, element_pool, cell_type, cutoff):

        AlloyGen = AlloysGen(element_pool, concentrations, cell_type)

        NN1 = coordination_numbers[cell_type][0]
        NN2 = coordination_numbers[cell_type][1]
        pop_fitness = []
        for configuration in configurations:
            CN1_list, CN2_list = AlloyGen.get_coordination_numbers(configuration, cutoff)
            # shell1_fitness_AA = self.get_shell1_fitness_AA(CN1_list)
            shell2_fitness_AA = self.get_shell2_fitness_AA(CN2_list, NN2)
            shell1_fitness_AB = self.get_shell1_fitness_AB(CN1_list, concentrations, NN1)
            max_diff_element_fitness = self.get_max_diff_element_fitness(max_diff_element, configuration)
            fitness = (
                # sum(shell1_fitness_AA.values())
                    +sum(max_diff_element_fitness.values())
                    + sum(shell1_fitness_AB.values())
                    + sum(shell2_fitness_AA.values())
            )

            pop_fitness.append(fitness)
        # logger.info('shell1_fitness_AA:{:9.4f} \t shell1_fitness_AB: {:9.4f}  \t  max_diff_element_fitness: {
        # :9.4f}'.format( sum(shell1_fitness_AA.values()), sum(shell1_fitness_AB.values()),
        # sum(max_diff_element_fitness.values())))

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
