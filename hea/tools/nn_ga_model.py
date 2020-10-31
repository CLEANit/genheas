
import numpy as np
import random
import pymatgen as pmg
import copy

from pymatgen import Element

import torch



class NnGa(object):
    """

    """

    def __init__(self, n_structures=10, rate=0.25, alpha=0.1):
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

    #  @profile
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
    def gen_structure(self, alloyatoms, out_vector, atoms_list_set):
        """

        """
        atms = copy.deepcopy(alloyatoms)
        atom_list = [np.random.choice(atoms_list_set, p=vec) for vec in out_vector]
        atms.set_chemical_symbols(atom_list)
        atomic_fraction = {}

        compo = pmg.Composition(atms.get_chemical_formula())
        for elm in atoms_list_set:
            atomic_fraction[elm] = compo.get_atomic_fraction(Element(elm))
        return atms, atomic_fraction

    def gen_structures(self, alloyatoms, out_vector, atoms_list_set):
        atms = copy.deepcopy(alloyatoms)

        out_vector = out_vector.flatten()
        out_vector = out_vector / len(out_vector)
        atm_list = atoms_list_set * len(atms)  # normalize to 1
        atom_list = np.random.choice(atm_list, len(atms), p=out_vector)
        atms.set_chemical_symbols(atom_list)

        atomic_fraction = {}
        compo = pmg.Composition(atms.get_chemical_formula())
        for elm in atoms_list_set:
            atomic_fraction[elm] = compo.get_atomic_fraction(Element(elm))
        return atms, atomic_fraction

    # @profile
    def fitness(self, fraction, target):
        """
        fractions is a dictionary with concentration of each atom
        target is a dictionary with the target concentration
        return the Root-mean-square deviation
        """

        atoms_list = list(target.keys())

        target_conc = [target[elm] for elm in atoms_list]
        target_conc = np.array(target_conc)

        conc = [fraction[elm] for elm in atoms_list]
        conc = np.array(conc)
        diff = np.abs(conc - target_conc)
        N = len(conc)
        return np.sqrt((diff * diff).sum() / N)

    # @profile
    def get_population_fitness(self, alloyatoms, policies, atoms_list_set, target):

        policies_obj_funct = []
        for policy in policies:
            structures = []
            fractions = []

            for i in range(len(policies)):
                struct, frac = self.gen_structure(alloyatoms, policy, atoms_list_set)
                fractions.append(frac)
                structures.append(struct)

            obj_funct = [self.fitness(comp, target) for comp in fractions]

            policies_obj_funct.append(np.average(np.array(obj_funct)))

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
        noise = np.random.uniform(-1, 1, individual.shape) * alpha
        noised_individual = individual + noise
        return noised_individual

    # @profile
    def make_next_generation(self, previous_population, key, rate=None):
        if rate is None:
            rate = self.rate

        sorted_by_fitness_population = self.sort_population_by_fitness(previous_population, key)
        population_size = len(previous_population)

        top_size = int(np.ceil(population_size * rate))

        # take top 25%
        top_population = sorted_by_fitness_population[:top_size]

        next_generation = top_population

        # randomly mutate the weights of the top 25% of structures to make up the remaining 75%
        selection = self.choice_random(top_population, (population_size - top_size))
        next_generation.extend(selection)

        return next_generation
