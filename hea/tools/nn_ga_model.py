
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
#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
coordination_numbers ={'fcc': [12,6], 'bcc':[8,6]}

class NnGa(object):
    """

    """

    def __init__(self,AlloyGen, max_diff_element,
                 n_structures=10, rate=0.25, alpha=0.1, device="cpu"):
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
        #self.element_pool = element_pool
        #self.concentration = concentration
        #self.cell_type= cell_type
        #self.cell_size = cell_size
        self.max_diff_element = max_diff_element
        self.AlloyGen =AlloyGen
        #AlloysGen.__init__(self,self.element_pool, self.concentration, self.cell_type, self.cell_size)
        # Then call the function of with self ( self.sites_neighbor_list)



    def gen_policies(self, Feedforward, input_size, output_size, input_tensor, nb_policies):
        policies = []  # list of output vectors
        policies_weights = []

        for i in range(nb_policies):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    def gen_structure_numpy(self, alloyatoms, output_tensor, element_pool):
        """
        gen structure using numpy.randomchoice
        """
        output_vector = output_tensor.cpu().detach().numpy()
        replace = True  # default
        atom_list = [np.random.choice(element_pool, p=vec, size=1, replace=replace) for vec in output_vector]
        alloyatoms.set_chemical_symbols(atom_list)
        atomic_fraction = {}

        compo = pmg.Composition(alloyatoms.get_chemical_formula())
        for elm in element_pool:
            atomic_fraction[elm] = compo.get_atomic_fraction(Element(elm))
            return alloyatoms, atomic_fraction

    def gen_structure(self, alloyatoms, output_tensor, element_pool=None, max_diff_element=None):
        """
        using pytorch
        """
        # replace = True  # default  False
        # atom_list = [element_pool[elem.multinomial(num_samples=1, replacement=replace)] for elem in output_tensor]
        # alloyatoms.set_chemical_symbols(atom_list)
        # compo = pmg.Composition(alloyatoms.get_chemical_formula())
        # fractional_composition = compo.fractional_composition
        if element_pool is None:
            element_pool = self.element_pool

        if max_diff_element is None:
            max_diff_element = self.max_diff_element


        replace = True # default  False
        elems_in =[]
        max_diff_elem = copy.deepcopy(max_diff_element)
        for iout, output in enumerate(output_tensor):

            choice = output.multinomial(num_samples=1, replacement=replace)
            atm = element_pool[choice]

            idx = element_pool.index(atm)
            while max_diff_elem[idx]== 0: #  We have the max of this elements
                prob = output [idx] # take the proba of the element
                prob = prob/(len(element_pool)-1)  #divide by nber element -1
                output = output+prob # add the value to each component
                output [idx] = 0 # set the selected proba to 0
                choice = output.multinomial(num_samples=1, replacement=replace)
                atm = element_pool[choice]
                idx = element_pool.index(atm)

            max_diff_elem[idx]-=1
            alloyatoms.symbols[iout] = atm
            elems_in.append(atm)

            #max_diff_elem = {x:elems_in.count(x) for x in elems_in}
        compo = pmg.Composition(alloyatoms.get_chemical_formula())
        fractional_composition = compo.fractional_composition

        return alloyatoms, fractional_composition

    # def gen_structures(self, alloyatoms, out_vector, element_pool):
    #     atms = copy.deepcopy(alloyatoms)

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
        return np.abs(np.log (np.sqrt((diff * diff).sum() / N)))



    def fitness2(self, configuration, combinasons, concentrations,lattice_param,cell_type, element_list ):
        pmg_structure = AseAtomsAdaptor.get_structure(configuration)
        all_neighbors_list = self.AlloyGen.sites_neighbor_list(pmg_structure, lattice_param)
        atomic_vec_array = self.AlloyGen.get_neighbors_type(all_neighbors_list)
        atm_numbers = configuration.numbers

        shells = np.hsplit(atomic_vec_array, [coordination_numbers[cell_type][0],atomic_vec_array.shape[0]])
        shell_1, shell_2 = shells[0], shells[1]
        occurrences_1 = {}
        occurrences_2 = {}
        for elm in element_list:
            occurrences_1[elm] = np.count_nonzero(shell_1 == atomic_numbers[elm] , axis=1)
            occurrences_2[elm] = np.count_nonzero(shell_2 == atomic_numbers[elm], axis=1)

        terms_1 = {}
        terms_2 = {}
        for conmbi in combinasons:
            atm1 = conmbi.split('-')[0]
            atm2 = conmbi.split('-')[1]
            terms_1[conmbi] =  occurrences_1[atm1][np.where(atm_numbers==atomic_numbers[atm2])]
            terms_2[conmbi] =  occurrences_2[atm1][np.where( atm_numbers==atomic_numbers[atm2])]

        fitn_1 = {}
        fitn_2 = {}
        neighbors_1 = {}
        neighbors_2 = {}

        for key in combinasons:
            atm1 = key.split('-')[0]
            #fitn_1[key] = np.exp(np.std(terms_1[key],ddof=1)/np.sqrt(len(terms_1[key])))
            #fitn_1[key] = np.exp(scipy.stats.sem(terms_1[key]))
            fitn_1[key] = np.exp(np.std(terms_1[key]))

            neighbors_1[key] = np.exp(np.abs(np.average(terms_1[key])/concentrations[atm1]
                                             -coordination_numbers[cell_type][0]))

            #fitn_2[key] = np.exp(np.std(terms_2[key],ddof=1)/np.sqrt(len(terms_2[key])))
            fitn_2[key] = np.exp(np.std(terms_2[key]))
            #coordination_2[key] = np.average(terms_2[key])
            neighbors_2[key] = np.exp(np.abs(np.average(terms_2[key])/concentrations[atm1]
                                             -coordination_numbers[cell_type][1]))

        # for key, val in terms_2.items():
        #     fitn_2[key] = np.exp(np.std(val))
        #     coordination_2[key] = np.average(val)

        fitness_CN1 = 0 # Cordination number
        fitness_CN2 = 0
        fitness_NN1 = 0 # Concentration
        fitness_NN2 = 0
        for key in combinasons:
            fitness_CN1 += fitn_1[key]/len(combinasons)
            fitness_CN2 += fitn_2[key]/len(combinasons)
            fitness_NN1 += neighbors_1[key]/len(combinasons)
            fitness_NN2 += neighbors_2[key]/len(combinasons)

        NN1 = {}
        NN2 = {}
        for key in combinasons:
            atm1 = key.split('-')[0]
            NN1[key] = np.average(terms_1[key])/concentrations[atm1]
            NN2[key] =np.average(terms_2[key])/concentrations[atm1]

        fitness_CN = np.abs(1-fitness_CN1)+ np.abs(1-fitness_CN1)
        fitness_NN= np.abs(1-fitness_NN1)+ np.abs(1-fitness_NN1)

        return fitness_CN , NN1, NN2


    # @profile
    def get_population_fitness(self, alloyatoms, policies, element_pool, concentrations,cell_type,lattice_param,combinasons):
        #logger.info('get_population_fitness')
        policies_obj_funct = []
        for policy in policies:
            structures = []
            #fractions = []
            obj_functs = []
            for i in range(len(policies)):
                struct, fractional_composition = self.gen_structure(alloyatoms, policy, element_pool)
                obj_funct2, shell1, shell2 = self.fitness2(struct, combinasons, concentrations,lattice_param,cell_type,  element_pool)
                #obj_funct1 = self.fitness1(fractional_composition, concentrations)
                #fractions.append(frac)
                structures.append(struct)
                obj_functs.append(obj_funct2)

            policies_obj_funct.append(np.average(np.array(obj_functs)))

        return policies_obj_funct

    def get_population_fitness1(self, alloyatoms, policies, element_pool, concentrations,cell_type,lattice_param,combinasons):
        #logger.info('get_population_fitness')
        policies_obj_funct = []
        for policy in policies:
            structures = []
            #fractions = []
            obj_functs = []
            for i in range(len(policies)):
                struct, fractional_composition = self.gen_structure(alloyatoms, policy, element_pool)
                #obj_funct2, shell1, shell2 = self.fitness2(struct, combinasons, concentrations,lattice_param,cell_type,  element_pool)
                obj_funct1 = self.fitness1(fractional_composition, concentrations)
                #fractions.append(frac)
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
        noise = torch.FloatTensor(individual.shape).uniform_(-1,1)*alpha
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
            sorted_by_fitness_population = self.sort_population_by_fitness(previous_population, key)
        else:
            sorted_by_fitness_population = previous_population
        population_size = len(previous_population)

        top_size = int(np.ceil(population_size * rate))

        # take top 25%
        top_population = sorted_by_fitness_population[:top_size]

        next_generation = top_population

        # randomly mutate the weights of the top 25% of structures to make up the remaining 75%
        selection = self.choice_random(top_population, (population_size - top_size))
        selections = [self.mutate(selec) for selec in selection]
        next_generation.extend(selections)

        return next_generation
