
import numpy as np
import random
import pymatgen as pmg
import copy
#import scipy

from pymatgen import Element
from pymatgen.io.ase import AseAtomsAdaptor
from hea.tools.alloysgen import AlloysGen
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

    def __init__(self,n_structures=10, rate=0.25, alpha=0.1, device="cpu"):
        """
        #nb_polocies: int : nber of polocies generated at each generation
        n_structures: int :  nber of structure generated for each policies
        rate: float : rate of policies to keep
        alpha: parameter for gene mutation
        """
        # self.nb_polocies = nb_polocies
        self.n_structures = n_structures
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

    # def gen_structures(self, alloyAtoms, out_vector, element_pool):
    #     atms = copy.deepcopy(alloyAtoms)

    #     out_vector = out_vector.flatten()
    #     out_vector = out_vector / len(out_vector)
    #     atm_list = element_pool * len(atms)  # normalize to 1
    #     atom_list = np.random.choice(atm_list, len(atms), p=out_vector)
    #     atms.set_chemical_symbols(atom_list)

    #     atomic_fraction = {}
    #     compo = pmg.Composition(atms.get_chemical_formula())
    #     for elm in element_pool:
    #         atomic_fraction[elm] = compo.get_atomic_fraction(Element(elm))
    #     return atms, atomic_fraction

    # @profile
    def fitness1(self, fraction, target):
        """
        fraction: is a dictionary with concentration of each atom
        target: is a dictionary with the target concentration
        return the Root-mean-square deviation
        """

        atoms_list = list(target.keys())

        target_conc = [target[elm] for elm in atoms_list]
        target_conc = np.array(target_conc)

        conc = [fraction[elm] for elm in atoms_list]
        conc = np.array(conc)
        diff = np.abs(conc - target_conc)
        N = len(conc)
        return np.abs(np.log(np.sqrt((diff * diff).sum() / N)))

    def fitness2(
            self,
            configuration,
            combinasons,
            concentrations,
            lattice_param,
            cell_type,
            element_list):
        pmg_structure = AseAtomsAdaptor.get_structure(configuration)
        all_neighbors_list = self.AlloyGen.sites_neighbor_list(
            pmg_structure, lattice_param)
        atomic_vec_array = self.AlloyGen.get_neighbors_type(all_neighbors_list)
        atm_numbers = configuration.numbers

        shells = np.hsplit(
            atomic_vec_array, [
                coordination_numbers[cell_type][0], atomic_vec_array.shape[0]])
        shell_1, shell_2 = shells[0], shells[1]
        occurrences_1 = {}
        occurrences_2 = {}
        for elm in element_list:
            occurrences_1[elm] = np.count_nonzero(
                shell_1 == atomic_numbers[elm], axis=1)
            occurrences_2[elm] = np.count_nonzero(
                shell_2 == atomic_numbers[elm], axis=1)

        terms_1 = {}
        terms_2 = {}
        for conmbi in combinasons:
            atm1 = conmbi.split('-')[0]
            atm2 = conmbi.split('-')[1]
            terms_1[conmbi] = occurrences_1[atm1][np.where(
                atm_numbers == atomic_numbers[atm2])]
            terms_2[conmbi] = occurrences_2[atm1][np.where(
                atm_numbers == atomic_numbers[atm2])]

        fitn_1 = {}
        fitn_2 = {}
        neighbors_1 = {}
        neighbors_2 = {}

        for key in combinasons:
            atm1 = key.split('-')[0]
            # fitn_1[key] = np.exp(np.std(terms_1[key],ddof=1)/np.sqrt(len(terms_1[key])))
            # fitn_1[key] = np.exp(scipy.stats.sem(terms_1[key]))
            fitn_1[key] = np.exp(np.std(terms_1[key]))

            neighbors_1[key] = np.exp(
                np.abs(
                    np.average(
                        terms_1[key]) /
                    concentrations[atm1] -
                    coordination_numbers[cell_type][0]))

            # fitn_2[key] = np.exp(np.std(terms_2[key],ddof=1)/np.sqrt(len(terms_2[key])))
            fitn_2[key] = np.exp(np.std(terms_2[key]))
            # coordination_2[key] = np.average(terms_2[key])
            neighbors_2[key] = np.exp(
                np.abs(
                    np.average(
                        terms_2[key]) /
                    concentrations[atm1] -
                    coordination_numbers[cell_type][1]))

        # for key, val in terms_2.items():
        #     fitn_2[key] = np.exp(np.std(val))
        #     coordination_2[key] = np.average(val)

        fitness_CN1 = 0  # Cordination number
        fitness_CN2 = 0
        fitness_NN1 = 0  # Concentration
        fitness_NN2 = 0
        for key in combinasons:
            fitness_CN1 += fitn_1[key] / len(combinasons)
            fitness_CN2 += fitn_2[key] / len(combinasons)
            fitness_NN1 += neighbors_1[key] / len(combinasons)
            fitness_NN2 += neighbors_2[key] / len(combinasons)

        NN1 = {}
        NN2 = {}
        for key in combinasons:
            atm1 = key.split('-')[0]
            NN1[key] = np.average(terms_1[key]) / concentrations[atm1]
            NN2[key] = np.average(terms_2[key]) / concentrations[atm1]

        fitness_CN = np.abs(1 - fitness_CN1) + np.abs(1 - fitness_CN1)
        fitness_NN = np.abs(1 - fitness_NN1) + np.abs(1 - fitness_NN1)

        return fitness_CN, NN1, NN2

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



    def get_neighbors_deviation(self,CN_list,concentrations, NNeighbours):
        """
        CN_list:list of dictionanry
        alloy_atoms: structure(Atoms class)
        NNeighbours:  ( number of neighbours in the shell)
        return a dictionary:


        """

        #NNeighbours = coordination_numbers[self.cell_type][0]

        fitness = {}
        #compo = pmg.Composition(alloyAtoms.get_chemical_formula())
        #fractional_composition = compo.fractional_composition

        for key, val in CN_list.items():
            atm = key.split('-')[0]

            target = NNeighbours*concentrations[atm]

            fitness[key] = self._rmsd(val,target )


        return fitness


    def get_shell_fitness(self,CN_list):
        """
         CN_list:list of dictionanry
        """
        fitness = {}
        target = 0

        for key, val in CN_list.items():
            atm1, atm2 = key.split('-')
            if atm1 == atm2:
                fitness[key] = self._rmsd(val,target )

        return fitness

    def get_offset_fitness(self,CN_list):
        """
       CN_list:list of dictionanry
       alloy_atoms: structre(Atoms class)
       NNeighbours:  ( number of neighbours in the shell)
        """



        fitness = {}
        target = 0

        for key, val in CN_list.items():
            atm1, atm2 = key.split('-')
            if atm1 == atm2:
                target = 0
                fitness[key] = self._rmsd(val,target )


        return fitness


    # @profile

    def get_population_fitness(self,alloyAtoms, concentrations, outputs,element_pool,cell_type,cutoff,combinasons):
        # logger.info('get_population_fitness')

        AlloyGen = AlloysGen(element_pool, concentrations, cell_type)



        #combinasons = self._combinaison(element_pool)


        max_diff_element = self.get_max_diff_elements(element_pool, concentrations, len(alloyAtoms))

        NNeighbours = coordination_numbers[cell_type][0]
        pop_fitness = []
        for output in outputs:
            structures = []
            # fractions = []
            fitness_list = []

            for i in range(len(outputs)):

                configuration, fractional_composition =  \
                    self.gen_structure_from_output(alloyAtoms,output,element_pool,max_diff_element)


                CN1_list, CNoffset0_list = AlloyGen.get_coordination_numbers(configuration,cutoff)
                shell_fitness = self.get_shell_fitness(CN1_list)
                offset_fitness = self.get_offset_fitness(CNoffset0_list)
                fitness =   sum(shell_fitness.values()) #+  sum(offset_fitness.values())

                #structures.append(struct)
                fitness_list.append(fitness)

            pop_fitness.append(np.average(np.array(fitness)))

        return pop_fitness

    def get_population_fitness1(
            self,
            alloyAtoms,
            policies,
            element_pool,
            concentrations,
            cell_type,
            lattice_param,
            combinasons):
        # logger.info('get_population_fitness')
        policies_obj_funct = []
        for policy in policies:
            structures = []
            # fractions = []
            obj_functs = []
            for i in range(len(policies)):
                struct, fractional_composition = self.gen_structure(
                    alloyAtoms, policy, element_pool)
                # obj_funct2, shell1, shell2 = self.fitness2(struct, combinasons, concentrations,lattice_param,cell_type,  element_pool)
                obj_funct1 = self.fitness1(
                    fractional_composition, concentrations)
                # fractions.append(frac)
                structures.append(struct)
                obj_functs.append(obj_funct1)

            policies_obj_funct.append(np.average(np.array(obj_functs)))

        return policies_obj_funct
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
