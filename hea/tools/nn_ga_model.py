
import numpy as np
import random
import pymatgen as pmg
import copy
#import scipy

from pymatgen import Element
from pymatgen.io.ase import AseAtomsAdaptor
from tools.alloysgen import AlloysGen
#import sites_neighbor_list, get_neighbors_type
import torch
from ase.data import atomic_numbers
import logging
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
coordination_numbers = {'fcc': [12, 6, 24], 'bcc': [8, 6, 12]}



class NnGa(object):
    """

    """

    def __init__(self, rate=0.25, alpha=0.1, device="cpu"):
        """
        #nb_polocies: int : nber of polocies generated at each generation
        n_structures: int :  nber of structure generated for each policies
        rate: float : rate of policies to keep
        alpha: parameter for gene mutation
        """
        # self.nb_polocies = nb_polocies

        self.rate = rate
        self.alpha = alpha
        self.device = device
        # self.element_pool = element_pool
        # self.concentration = concentration
        # self.cell_type= cell_type
        # self.cell_size = cell_size
        #self.max_diff_element = max_diff_element
        # AlloysGen.__init__(self,self.element_pool, self.concentration, self.cell_type, self.cell_size)
        # Then call the function of with self ( self.sites_neighbor_list)

    def _combinaison(self, element_pool):
        """
        combinasons list of  unique pair of atom in the structure
        """
        combinasons = []
        for i in range(len(element_pool)):
            for j in range(len(element_pool)):
                combinasons.append(element_pool[i]+'-'+element_pool[j])
        return combinasons

    def get_max_diff_elements(self,element_pool, concentrations, nb_atoms):
            """
                compute the maximun number of different element base on concentration
                and the total number of atoms
            """
            max_diff = []
            for elm in element_pool:
                max_diff.append(round(nb_atoms * concentrations[elm]))
            if sum(max_diff) == nb_atoms:
                self.max_diff_element = max_diff
                return self.max_diff_element
            elif sum(max_diff) < nb_atoms:
                ielem = 1
                while sum(max_diff) != nb_atoms or ielem < len(max_diff):
                    max_diff[ielem] = max_diff[ielem] + 1
                    ielem += 1
                self.max_diff_element = max_diff
                return self.max_diff_element

    def gen_policies(
            self,
            Feedforward,
            input_size,
            output_size,
            input_tensor,
            nb_policies):
        policies = []  # list of output vectors
        policies_weights = []

        for i in range(nb_policies):
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            model = Feedforward(input_size, output_size).to(device)
            input_tensor = input_tensor.to(device)
            output_tensor = model(input_tensor)
            output_vector = output_tensor.cpu().detach().numpy()
            policies.append(output_vector)

            init_weight = copy.deepcopy(model.l1.weight.data)
            init_weight = init_weight.cpu().detach().numpy()
            policies_weights.append(init_weight)
        return policies, policies_weights

    # @profile
    def gen_structure_numpy(self, alloyAtoms, output_tensor, element_pool):
        """
        gen structure using numpy.randomchoice
        """
        output_vector = output_tensor.cpu().detach().numpy()
        replace = True  # default
        atom_list = [
            np.random.choice(
                element_pool,
                p=vec,
                size=1,
                replace=replace) for vec in output_vector]
        alloyAtoms.set_chemical_symbols(atom_list)
        atomic_fraction = {}

        compo = pmg.Composition(alloyAtoms.get_chemical_formula())
        for elm in element_pool:
            atomic_fraction[elm] = compo.get_atomic_fraction(Element(elm))
            return alloyAtoms, atomic_fraction

    def gen_structure_from_output(self, template, output_tensor, element_pool, max_diff_element):
        """
        using pytorch
        """

        replace = True  # default  False
        elems_in = []
        max_diff_elem = copy.deepcopy(max_diff_element)
        for iout, output in enumerate(output_tensor):

            choice = output.multinomial(num_samples=1, replacement=replace)
            atm = element_pool[choice]

            idx = element_pool.index(atm)
            while max_diff_elem[idx] == 0:  # We have the max of this elements
                prob = output[idx]  # take the proba of the element
                # divide by nber element -1
                prob = prob / (len(element_pool) - 1)
                output = output + prob  # add the value to each component
                output[idx] = 0  # set the selected proba to 0
                choice = output.multinomial(num_samples=1, replacement=replace)
                atm = element_pool[choice]
                idx = element_pool.index(atm)

            max_diff_elem[idx] -= 1
            template.symbols[iout] = atm
            elems_in.append(atm)

            # max_diff_elem = {x:elems_in.count(x) for x in elems_in}
        compo = pmg.Composition(template.get_chemical_formula())
        fractional_composition = compo.fractional_composition

        return template, fractional_composition



    # # @profile
    # def fitness1(self, fraction, target):
    #     """
    #     fraction: is a dictionary with concentration of each atom
    #     target: is a dictionary with the target concentration
    #     return the Root-mean-square deviation
    #     """

    #     atoms_list = list(target.keys())

    #     target_conc = [target[elm] for elm in atoms_list]
    #     target_conc = np.array(target_conc)

    #     conc = [fraction[elm] for elm in atoms_list]
    #     conc = np.array(conc)
    #     diff = np.abs(conc - target_conc)
    #     N = len(conc)
    #     return np.abs(np.log(np.sqrt((diff * diff).sum() / N)))



    def _rmsd(self, V, W):
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
        diff = np.abs(np.array(V) - np.array(W))
        N = len(V)
        return np.sqrt((diff * diff).sum() / N)

    def get_shell1_fitness_AA(self,CN1_list):

        """
        Minimize the $N_{aa}$ in the first coordination shell
         CN_list:list of dictionanry
        """
        fitness = {}
        target = 0

        for key, val in CN1_list.items():
            atm1, atm2 = key.split('-')
            if atm1 == atm2:
                fitness[key] = self._rmsd(val,target )

        return fitness

    def get_shell2_fitness_AA(self,CN2_list,NNeighbours):

        """
         Maximize the $N_{aa}$ in the second coordination shell
         CN_list:list of dictionanry
        """
        fitness = {}
        target = NNeighbours

        for key, val in CN2_list.items():
            atm1, atm2 = key.split('-')
            if atm1 == atm2:
                fitness[key] = self._rmsd(val,target )

        return fitness

    def get_shell1_fitness_AB(self,CN1_list,concentrations, NNeighbours):
        """
        Homogenize the $N_{a b}$ in the first coordination shell

        CN_list:list of dictionanry
        alloy_atoms: structure(Atoms class)
        NNeighbours:  ( number of neighbours in the shell)
        return a dictionary:


        """



        fitness = {}
        nb_species = len(concentrations.keys())

        for key, val in CN1_list.items():
            atm1 ,atm2 = key.split('-')
            if atm1 != atm2:
                param = NNeighbours*concentrations[atm2]
                target = NNeighbours*concentrations[atm1] + param/(nb_species-1)

                fitness[key] = self._rmsd(val,target )


        return fitness








    def get_max_diff_element_fitness(self, max_diff_element, alloyAtoms):
        """
        max_diff_element : maximun number of each specie

        """
        element_list = alloyAtoms.get_chemical_symbols()

        fitness = {}
        max_diff_elem = {x:element_list.count(x) for x in element_list}

        for key, val in max_diff_elem.items():

            fitness[key] = self._rmsd([val],[max_diff_element[key]] )

        return fitness

    # @profile

    def get_population_fitness(self,configurations, concentrations,
                               max_diff_element,element_pool,cell_type,cutoff,combinasons):
        # logger.info('get_population_fitness')

        AlloyGen = AlloysGen(element_pool, concentrations, cell_type)

        NN1 = coordination_numbers[cell_type][0]
        NN2 = coordination_numbers[cell_type][1]
        pop_fitness = []
        for configuration in configurations:



            CN1_list, CN2_list = AlloyGen.get_coordination_numbers(configuration,cutoff)
            shell1_fitness_AA = self.get_shell1_fitness_AA(CN1_list)
            #shell2_fitness_AA= self.get_shell2_fitness_AA(CN2_list, NN2)
            shell1_fitness_AB= self.get_shell1_fitness_AB(CN1_list,concentrations, NN1)
            max_diff_element_fitness =  self.get_max_diff_element_fitness(max_diff_element, configuration)
            fitness =   sum(shell1_fitness_AA.values())  +sum(shell1_fitness_AB.values())+sum(max_diff_element_fitness.values()) #+sum(shell2_fitness_AA.values())

            pop_fitness.append(fitness)



        return np.average(np.array(pop_fitness))


    # @profile

    def sort_population_by_fitness(self, population, key):
        sorted_index = np.argsort(key)
        sorted_population = [population[idx] for idx in sorted_index]
        return sorted_population

    # @profile
    def choice_random(self, population, n):
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
        """
        if rate is None:
            rate = self.rate

        if key is not None:
            sorted_by_fitness_population = self.sort_population_by_fitness(
                previous_population, key)
        else:
            sorted_by_fitness_population = previous_population
        population_size = len(previous_population)

        top_size = int(np.ceil(population_size * rate))

        # take top 25%
        top_population = sorted_by_fitness_population[:top_size]

        next_generation = top_population

        # randomly mutate the weights of the top 25% of structures to make up
        # the remaining 75%
        selection = self.choice_random(
            top_population, (population_size - top_size))
        selections = [self.mutate(selec) for selec in selection]
        next_generation.extend(selections)

        return next_generation
