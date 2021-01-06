import copy
import os
import random

import numpy as np
import pymatgen as pmg
import torch
import yaml
from ase.build import bulk
from ase.lattice.cubic import BodyCenteredCubic, FaceCenteredCubic
from clusterx.parent_lattice import ParentLattice
from clusterx.structures_set import StructuresSet
from clusterx.super_cell import SuperCell
from pymatgen import Element
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.transformations.site_transformations import \
    ReplaceSiteSpeciesTransformation
from pyxtal.crystal import random_crystal
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable

from hea.tools.log import logger

coordination_numbers = {'fcc': [12, 6, 24], 'bcc': [8, 6, 12]}

properties_list = ['number', 'group', 'row', 'is_metal', 'is_transition_metal',
                   'is_alkali', 'is_alkaline', 'is_metalloid', 'atomic_radius',
                   'oxidation_states', 'VEC', 'electronegativity']


class AlloysGen(object):

    def __init__(self, element_pool, concentration, crystalstructure, oxidation_states=None):
        self.element_pool = element_pool
        self.concentration = concentration
        self.crystalstructure = crystalstructure
        self.oxidation_states = oxidation_states
        self.combination = self.get_combination(self.element_pool)
        self.input_size = None

    @staticmethod
    def get_combination(element_pool):
        """
        combinations list of  unique pair of atom in the structure
        """
        combinations = []
        for i in range(len(element_pool)):
            for j in range(len(element_pool)):
                combinations.append(element_pool[i] + '-' + element_pool[j])
        return combinations

    @staticmethod
    def gen_alloy_supercell(element_pool, concentrations, crystalstructure, size, lattice_param=None):
        """
        :param lattice_param:
        :param element_pool: list of element in the alloy
        :param concentrations:
        :param crystalstructure: fcc or bcc
        :param size:
        :return: Alloy supercell
        """
        if lattice_param is None:
            lattice_param = [None, None, None]
        prim = []
        if crystalstructure == 'fcc':
            if all(elem is None for elem in lattice_param):
                try:
                    lattice_param = FaceCenteredCubic(element_pool[0]).cell.cellpar()[:3]
                except Exception as err:
                    print(err)
                    raise Exception('Please provide the lattice parameter')

            a = lattice_param[0]
            prime = bulk(name='X', crystalstructure='fcc', a=a, b=None, c=None)
            for elm in element_pool:
                prime_copy = copy.deepcopy(prime)
                prime_copy.set_chemical_symbols(elm)

                prim.append(prime_copy)
        elif crystalstructure == 'bcc':
            if all(elem is None for elem in lattice_param):
                try:
                    '''# array([ 3.52,  3.52,  # 3.52, 90.  , 90.  , 90.  ]) '''
                    lattice_param = BodyCenteredCubic(element_pool[0]).cell.cellpar()[:3]

                except Exception as err:
                    print(err)
                    raise Exception('Please provide the lattice parameter')

            a = lattice_param[0]
            prime = bulk(name='X', crystalstructure='bcc', a=a, b=None, c=None)

            for elm in element_pool:
                prime_copy = copy.deepcopy(prime)
                prime_copy.set_chemical_symbols(elm)

                prim.append(prime_copy)

        elif crystalstructure == 'hpc':

            # if  all(elem  is None for elem in lattice_param):
            #     try:
            #         a =BodyCenteredCubic(element_pool[0]).cell[0][0]
            #         b =BodyCenteredCubic(element_pool[0]).cell[0][0]
            #         c =BodyCenteredCubic(element_pool[0]).cell[0][0]
            #     except Exception as err:
            #         print(err)
            #         raise Exception ('Please provide the lattice parameter')
            # a=None
            # b=None
            # c=None

            # for elm in element_pool:
            #     #prim.append(BodyCenteredCubic(elm))
            #     prim.append(bulk(name=elm, crystalstructure='hpc', a=a, b=b, c=c))
            raise Exception('hpc is not yet implemented')
        else:
            raise Exception('Only fcc abd bcc are defined')

        platt = ParentLattice(prim[0], substitutions=prim[1:])
        scell = SuperCell(platt, size)

        # lattice_param = prim[0].cell[0][0]
        sset = StructuresSet(platt)
        nstruc = 1
        nb_atm = []
        sub = {}
        for elm in element_pool:
            nb_atm.append(round(len(scell) * concentrations[elm]))
        if sum(nb_atm) == len(scell):
            sub[0] = nb_atm[1:]
            for _ in range(nstruc):
                sset.add_structure(scell.gen_random(sub))
        elif sum(nb_atm) < len(scell):
            ielem = 1
            while sum(nb_atm) != len(scell) or ielem < len(nb_atm):
                nb_atm[ielem] = nb_atm[ielem] + 1
                ielem += 1
            sub[0] = nb_atm[1:]
            for _ in range(nstruc):
                sset.add_structure(scell.gen_random(sub))
        else:
            raise Exception(' Sum of concentrations is not equal to 1')

        clx_structure = sset.get_structures()[0]
        alloyAtoms = clx_structure.get_atoms()  # ASE Atoms Class
        # alloyStructure = AseAtomsAdaptor.get_structure(alloyAtoms)  # Pymatgen Structure
        # alloyComposition = pmg.Composition(alloyAtoms.get_chemical_formula())  # Pymatgen Composition
        return alloyAtoms, lattice_param

    @staticmethod
    def gen_raw_crystal(crystalstructure, size, element='X', lattice_param=None):
        """
        generate a crystal filled with  element: 'element'
        """

        if crystalstructure == 'fcc':
            if lattice_param is None:
                # if all(elem is None for elem in lattice_param):
                raise Exception('Please provide the lattice parameter')

            a = lattice_param[0]
            prime = bulk(name=element, crystalstructure='fcc', a=a, b=None, c=None)
            crystal = prime * tuple(size)

        elif crystalstructure == 'bcc':
            if all(elem is None for elem in lattice_param):
                raise Exception('Please provide the lattice parameter')

            a = lattice_param[0]
            prime = bulk(name=element, crystalstructure='bcc', a=a, b=None, c=None)
            crystal = prime * tuple(size)
        elif crystalstructure == 'hpc':

            raise Exception('hpc is not yet implemented')
        else:
            raise Exception('Only fcc abd bcc are implemented')

        return AseAtomsAdaptor.get_structure(crystal)

    @staticmethod
    def gen_random_crystal_3D(crystalstructure, element, size):
        """
        use the Pyxtal package to gererate a ramdom 3D crystal

        """
        if crystalstructure == 'fcc':
            try:
                metallic_crystal = random_crystal(225, element, size, 1.0, tm="metallic")
            except Exception as e:
                raise Exception(e)
        elif crystalstructure == 'bcc':
            try:
                metallic_crystal = random_crystal(216, element, size, 1.0, tm="metallic")
            except Exception as e:
                raise Exception(e)
        return metallic_crystal.to_pymatgen()

    @staticmethod
    def get_max_diff_elements(element_pool, concentrations, nb_atm):
        """
                compute the maximun number of different element base on concentration
                and the total number of atoms
            """
        max_diff = {}
        for elm in element_pool:
            max_diff[elm] = round(nb_atm * concentrations[elm])

        if sum(max_diff.values()) == nb_atm:
            return max_diff
        elif sum(max_diff.values()) < nb_atm:
            for elm in element_pool:
                while sum(max_diff.values()) != nb_atm:
                    max_diff[elm] = max_diff[elm] + 1

            return max_diff

    @staticmethod
    def sites_neighbor_list(structure, cutoff, site_number=None):
        """
        structure :  pymatgen structure class

        cutoff : distance cutoff

        return list of numpy array with the neighbors of each site


        center_indices, points_indices, offset_vectors, distances
        0  1  [ 0.  0. -1.] 2.892
        0  2  [ 0. -1.  0.] 2.892
        0  5  [ 0.  0. -1.] 2.892
        0  6  [ 0. -1.  0.] 2.892
        0  4  [-1.  0.  0.] 2.892
        0  3  [ 0.  0. -1.] 2.892
        0  6  [-1.  0.  0.] 2.892
        0  3  [ 0. -1.  0.] 2.892
        0  5  [-1.  0.  0.] 2.892
        0  4  [0. 0. 0.]    2.892
        0  2  [0. 0. 0.]    2.892
        0  1  [0. 0. 0.]    2.892

        """
        center_indices, points_indices, offset_vectors, distances = structure.get_neighbor_list(cutoff + 0.5)
        all_neighbors_list = []

        # all_distance_list = []
        for i in range(structure.num_sites):
            site_neighbor = points_indices[np.where(center_indices == i)]
            neighbor_distance = distances[np.where(center_indices == i)]
            inds = neighbor_distance.argsort()
            sortedNeighbor = site_neighbor[inds]
            # sortedDistance = neighbor_distance[inds]

            sortedNeighbor = np.insert(sortedNeighbor, 0, i)
            # sortedDistance=np.insert(sortedDistance, 0, 0)
            all_neighbors_list.append(sortedNeighbor)
            # all_distance_list.append(sortedDistance)

        if site_number is not None:
            return all_neighbors_list[site_number]
        else:
            return all_neighbors_list

    @staticmethod
    def site_neighbor_list(structure, cutoff, site_number):
        """
        structure :  pymatgen structure class

        cutoff : distance cutoff

        return list  neighbors of the site "site_number"
        """

        sites = structure.sites
        sites_neighbours = structure.get_neighbors(sites[site_number], cutoff + 0.5)
        neighbours_list = [sites[site_number].species_string]
        neighbours_list.extend([neighbour.species_string for neighbour in sites_neighbours])

        neighbours_list = ['X' if elm == 'X0+' else elm for elm in neighbours_list]

        return neighbours_list

    @staticmethod
    def get_neighbor_in_offset_zero(structure, cutoff):
        """
        structure :  pymatgen structure class

        cutoff : distance cutoff

        return list of neighbour in offset [0, 0, 0]
        """
        center_indices, points_indices, offset_vectors, distances = structure.get_neighbor_list(cutoff * 3 / 4)

        offset_list = []
        # all_distance_list = []
        for i in range(structure.num_sites):
            site_neighbor = points_indices[np.where(center_indices == i)]
            offset = offset_vectors[np.where(center_indices == i)]
            inds = [ielem for ielem, elem in enumerate(offset) if np.all(elem == 0)]

            offset_list.append(site_neighbor[inds])

        return offset_list

    @staticmethod
    def get_neighbors_type(all_neighbors_list, alloy_atoms):
        """
        neighbors_list: list of list
        return  atomic numbers and symbols of the neighbours list
        considered site is  not include in the list
        """

        atomic_numbers = alloy_atoms.numbers
        # symbols = alloyAtoms.symbols
        symbols = alloy_atoms.get_chemical_symbols()
        numbers_vec = []
        symbols_vec = []

        for nb_list in all_neighbors_list:
            numbers_vec.append(
                [atomic_numbers[i] for i in nb_list[1:]])  # exclude the fisrt atom because it is the site considered
            symbols_vec.append([symbols[i] for i in nb_list[1:]])
        return np.array(numbers_vec), np.array(symbols_vec)

    @staticmethod
    def get_neighbor_type(neighbors_list, alloy_atoms):
        """
        neighbors_list: list
        return  atomic numbers and symbols of the neighbours list
        """
        atomic_numbers = alloy_atoms.numbers
        symbols = alloy_atoms.get_chemical_symbols()

        numbers_vec = [atomic_numbers[i] for i in
                       neighbors_list[1:]]  # exclude the fisrt atom because it is the site considered
        symbols_vec = [symbols[i] for i in neighbors_list[1:]]
        return np.array(numbers_vec), np.array(symbols_vec)

    @staticmethod
    def get_neighbors_type2(self, all_neighbors_list, alloy_atoms):
        """
        neighbors_list: list of list
        return  atomic number and symbol of the neighbours list
        The considered site is  not include in the list
        """
        atomic_numbers = alloy_atoms.numbers

        symbols = alloy_atoms.get_chemical_symbols()
        numbers_vec = []
        symbols_vec = []

        for nb_list in all_neighbors_list:
            numbers_vec.append([atomic_numbers[i] for i in nb_list])
            symbols_vec.append([symbols[i] for i in nb_list])
        return numbers_vec, symbols_vec

    @staticmethod
    def get_neighbor_type2(neighbors_list, alloy_atoms):
        """
        neighbors_list: list
        return  atomic number and symbol of the neighbours list
        """
        atomic_numbers = alloy_atoms.numbers

        symbols = alloy_atoms.get_chemical_symbols()

        numbers_vec = [atomic_numbers[i] for i in neighbors_list]
        symbols_vec = [symbols[i] for i in neighbors_list]
        return np.array(numbers_vec), np.array(symbols_vec)

    @staticmethod
    def count_occurence_to_dict(arr, element_pool):
        """
        count occurence in numpy array and
        return a list dictinnary
        """
        my_list = []
        try:
            lenght = arr.shape[0]
        except Exception:
            lenght = len(arr)
        for i in range(lenght):
            unique, counts = np.unique(arr[i], return_counts=True)
            my_dict = dict(zip(unique, counts))
            for elem in element_pool:
                if elem not in my_dict.keys():
                    my_dict[elem] = 0
            my_list.append(my_dict)
        return my_list

    def count_neighbors_by_shell(self, neighbors_list, alloy_atoms, element_pool):
        """
        retrun list of dictionary


         for each atom the number of  nearest neighbor of each type
         in the first (12) and second (6) shells for fcc

         NNeighbours number of first nearest neighbours

         return list of dictionanry with number of neighbour of each type

        [{'Ag': 4, 'Pd': 8},  # first atom
         {'Ag': 6, 'Pd': 6},  # second atom
         {'Ag': 6, 'Pd': 6},
         {'Ag': 8, 'Pd': 4},
         {'Ag': 8, 'Pd': 4},
         {'Ag': 6, 'Pd': 6},
         {'Ag': 6, 'Pd': 6},
         {'Ag': 4, 'Pd': 8}]   # ....

        """

        NNeighbours = coordination_numbers[self.crystalstructure][0]

        numbers_vec, symbols_vec = self.get_neighbors_type(neighbors_list, alloy_atoms)

        shells = np.hsplit(symbols_vec, [NNeighbours])  # split into 12 and 6 for fcc
        # shellss=np.hsplit(shells[1], [6])

        shell1 = self.count_occurence_to_dict(shells[0], element_pool)  # first 12 column
        shell2 = self.count_occurence_to_dict(shells[1], element_pool)  # lat 6 Column

        return shell1, shell2

    @staticmethod
    def get_symbols_indexes(alloy_atoms):
        """
        retrun dictionanry with indexes of each type of atom
        """
        chemical_symbols = alloy_atoms.get_chemical_symbols()
        element_pool = list(set(chemical_symbols))

        symbols_indexes = {}

        for elem in element_pool:
            symbols_indexes[elem] = [i for i, x in enumerate(chemical_symbols) if x == elem]
        return symbols_indexes

    def _a_around_b(self, a, b, shell, alloy_atoms):
        """
        shell: list of dictionanry with the neighbour of each atom
        symbols_indexes : dictionary with the indexes of each element in the structure
        a: symbol of element a
        b: symbol of element b
        return list of number of atom a aound b
        """
        symbols_indexes = self.get_symbols_indexes(alloy_atoms)
        return [shell[i][a] for i in symbols_indexes[b]]

    def get_CN_list(self, shell, alloy_atoms):
        """
        combinations list of  unique pair of atom in the structure
        symbols_indexes : dictionary with the indexes of each element in the structure
        return dictionannry with the list of neighbor of each atom by type
        {'Ag-Ag': [6, 4, 4, 6], # Ag around Ag
         'Ag-Pd': [8, 6, 6, 8], # Ag around Pd
         'Pd-Ag': [6, 8, 8, 6],
         'Pd-Pd': [4, 6, 6, 4]}
        """
        CN_list = {}

        for combi in self.combination:
            atm1, atm2 = combi.split('-')
            try:
                CN_list[combi] = self._a_around_b(atm1, atm2, shell, alloy_atoms)
            except KeyError:
                CN_list[combi] = [0]
            except Exception as e:
                raise Exception(e)
        return CN_list

    def get_coordination_numbers(self, alloy_atoms, cutoff):
        """

        """

        alloyStructure = AseAtomsAdaptor.get_structure(alloy_atoms)
        # generate neighbor list in offset [ 0, 0, 0]
        all_neighbors_list = self.sites_neighbor_list(alloyStructure, cutoff)

        shell1, shell2 = self.count_neighbors_by_shell(all_neighbors_list, alloy_atoms, self.element_pool)

        CN1_list = self.get_CN_list(shell1, alloy_atoms)  # Coordination number
        CN2_list = self.get_CN_list(shell2, alloy_atoms)

        return CN1_list, CN2_list

    @staticmethod
    def get_atom_properties(element_pool, oxidation_states=None):
        """
            oxidation_state: dictionay with oxydation state of element

            return : dictionary with properties of each atom

            """

        loc = os.path.dirname(os.path.abspath(__file__))
        try:
            with open(os.path.join(loc, "data/VEC.yml"), "r") as fr:
                VEC = yaml.safe_load(fr)

        except Exception as e:
            raise Exception(e)

        properties = {}

        for el in element_pool:
            elm = Element(el)
            props = []
            for prop in properties_list:
                if prop == 'oxidation_states':
                    if oxidation_states is not None:
                        try:
                            props.append(0)
                            # props.append(oxidation_states[elm.name])
                        except Exception as e:
                            logger.error(f'{e}')
                            logger.warning(f'oxidation of {elm.name} set to 0')
                            props.append(0)
                    else:
                        props.append(0)
                elif prop == 'VEC':
                    try:
                        props.append(VEC[elm.name])  # valence

                    except Exception as e:
                        raise Exception(e)
                elif prop == 'electronegativity':
                    try:
                        props.append(elm.X)  # electronegativity
                    except Exception as e:
                        raise Exception(e)
                else:
                    try:
                        props.append(getattr(elm, prop))
                    except Exception as e:
                        raise Exception(e)

            properties[elm.name] = props

        properties['X'] = [0] * len(properties[element_pool[0]])
        return properties

    @staticmethod
    def get_input_vectors(all_neighbors_list, properties, alloy_structure):
        """
        all_neighbors_list: list of array with the list the neighbors  of each atom
        properties: dictionary with properties of each atom type
        """

        species = np.array(alloy_structure.species)

        input_vectors = []
        scaler = StandardScaler()
        for nb_list in all_neighbors_list:
            input_vector = []
            for elm in species[nb_list]:
                prop = properties[elm.name]
                input_vector.append(prop)
            input_vector = scaler.fit_transform(input_vector)
            input_vectors.append(input_vector)
        return np.array(input_vectors)

    @staticmethod
    def get_input_vector(neighbors_list, properties):
        """
        neighbors_list: list of array with the list the neighbors  an atom
        properties: dictionary with properties of each atom type
        """

        scaler = StandardScaler()

        input_vector = [properties[elm] for elm in neighbors_list]
        input_vector = scaler.fit_transform(input_vector)
        return np.array(input_vector)

    @staticmethod
    def _apply_constraint(output, pool_element, idex):
        replace = True
        prob = output[idex]  # take the proba of the element
        prob = prob / (len(pool_element) - 1)  # divide by nber element -1
        output = output + prob  # add the value to each component
        output[idex] = 0  # set the selected proba to 0
        # choice = torch.argmax(output_tensor)
        choice = output.multinomial(num_samples=1, replacement=replace)
        atm = pool_element[choice]
        # output_vector = output_tensor.to(device).numpy()
        # atm = np.random.choice(element_pool, p=output_vector)
        # pool_element.index(atm)
        return atm, output

    def gen_configuration(self, structureX, element_pool, cutoff, model, device,
                          max_diff_element=None, constrained=False):
        """
            add a contrain to the for the max_diff_element

            structureX: empty structure to be filled ( N sites)
            element_pool : list of dieffernet species
            max_diff_element: maximum nunber of different species
            model: NN model to train
            return generated Atoms configuration
            """

        # global output_tensor, max_diff_elem
        properties = self.get_atom_properties(element_pool, oxidation_states=self.oxidation_states)
        config = copy.deepcopy(structureX)
        replace = True  # default  False
        elems_in = []

        Natoms = structureX.num_sites

        if max_diff_element is not None:
            if sum(max_diff_element.values()) != Natoms:
                raise Exception('The number of site in the structure is not equal to sum of different elements')
            else:
                max_diff_elem = copy.deepcopy(max_diff_element)

        for i in range(Natoms):
            if i == 0:
                atm = random.choice(element_pool)
            else:
                neighbors_list = self.site_neighbor_list(config, cutoff, i)
                # atomX = AseAtomsAdaptor.get_atoms(config)
                # numbers_vec, symbols_vec = AlloyGen.get_neighbor_type2(
                #    neighbors_list, atomX)

                input_vector = self.get_input_vector(neighbors_list, properties)
                X_tensor = torch.from_numpy(input_vector).float()
                input_tensor = Variable(X_tensor, requires_grad=False).to(device)
                with torch.set_grad_enabled(False):
                    output_tensor = model(input_tensor)

                # choice = torch.argmax(output_tensor)
                choice = output_tensor.multinomial(num_samples=1, replacement=replace)

                atm = element_pool[choice]
            idx = element_pool.index(atm)

            if constrained and len(elems_in) > 1:

                atm_1 = elems_in[-1]
                atm_2 = elems_in[-2]
                idx = element_pool.index(atm)
                while atm_1 == atm_2 == atm:  # We have the max of this elements
                    atm, _ = self._apply_constraint(output_tensor, element_pool, idx)

            if constrained and max_diff_element is not None:
                idx = element_pool.index(atm)
                while max_diff_elem[atm] == 0:  # We have the max of this elements
                    atm, output_tensor= self._apply_constraint(output_tensor, element_pool, idx)

                max_diff_elem[atm] -= 1

            replace_species = ReplaceSiteSpeciesTransformation(
                indices_species_map={i: atm})
            config = replace_species.apply_transformation(config)

            elems_in.append(atm)

        # init_weight = copy.deepcopy(model.l1.weight.data)
        atms = AseAtomsAdaptor.get_atoms(config)
        # compo = pmg.Composition(atms.get_chemical_formula())
        # fractional_composition = compo.fractional_composition

        return atms
