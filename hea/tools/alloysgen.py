import yaml
import numpy as np
from ase.lattice.cubic import FaceCenteredCubic, BodyCenteredCubic
from ase.build import bulk
import pymatgen as pmg
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen import Element
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structures_set import StructuresSet
from pyxtal.crystal import random_crystal, random_crystal_2D
#from pyxtal import pyxtal
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.utils import to_categorical
import copy
from torch.autograd import Variable
import torch.nn.init as init
from pymatgen.transformations.site_transformations import ReplaceSiteSpeciesTransformation
import torch
import random
import os
coordination_numbers ={'fcc': [12,6,24], 'bcc':[8,6,12]}

class AlloysGen(object):

    def __init__(self, element_pool, concentration, crystalstructure):
        self.element_pool = element_pool
        self.concentration = concentration
        self.crystalstructure = crystalstructure
        # self.cell_size = cell_size
        # self.species = None
        # self.alloyAtoms = None
        # self.alloyStructure = None
        # self.alloyComposition = None
        self.combination = self.get_combination(self.element_pool)
        # self.max_diff_element = None



    def get_combination(self, element_pool):
        """
        combinations list of  unique pair of atom in the structure
        """
        combinations = []
        for i in range(len(element_pool)):
            for j in range(len(element_pool)):
                combinations.append(element_pool[i]+'-'+element_pool[j])
        return combinations






    def gen_alloy_supercell(self, element_pool, concentrations, crystalstructure, size,
                            lattice_param = [None, None, None]):
        """
        :param element_pool: list of element in the alloy
        :param concentrations:
        :param crystalstructure: fcc or bcc
        :param size:
        :return: Alloy supercell
        """
        prim = []
        if crystalstructure== 'fcc':
            if all(elem  is None for elem in lattice_param):
                try:
                    lattice_param = FaceCenteredCubic(element_pool[0]).cell.cellpar()[:3]
                except Exception as err:
                    print(err)
                    raise Exception ('Please provide the lattice parameter')

            a= lattice_param[0]
            prime = bulk(name='X', crystalstructure='fcc', a=a, b=None, c=None)
            for elm in element_pool:
                prime_copy =  copy.deepcopy(prime)
                prime_copy.set_chemical_symbols(elm)

                prim.append(prime_copy)
        elif crystalstructure== 'bcc':
            if all(elem  is None for elem in lattice_param):
                try:
                     lattice_param = BodyCenteredCubic(element_pool[0]).cell.cellpar()[:3]  # array([ 3.52,  3.52,  3.52, 90.  , 90.  , 90.  ])
                except Exception as err:
                    print(err)
                    raise Exception ('Please provide the lattice parameter')


            a = lattice_param[0]
            prime = bulk(name='X', crystalstructure='bcc', a=a, b=None, c=None)

            for elm in element_pool:
                prime_copy =  copy.deepcopy(prime)
                prime_copy.set_chemical_symbols(elm)

                prim.append(prime_copy)

        elif crystalstructure== 'hpc':

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

        #lattice_param = prim[0].cell[0][0]
        sset = StructuresSet(platt)
        nstruc = 1
        nb_atm = []
        sub = {}
        for elm in element_pool:
            nb_atm.append(round(len(scell) * concentrations[elm]))
        if sum(nb_atm) == len(scell):
            sub[0] = nb_atm[1:]
            for i in range(nstruc):
                sset.add_structure(scell.gen_random(sub))
        elif sum(nb_atm) < len(scell):
            ielem = 1
            while sum(nb_atm) != len(scell) or ielem < len(nb_atm):
                    nb_atm[ielem] = nb_atm[ielem] +1
                    ielem+=1
            sub[0] = nb_atm[1:]
            for i in range(nstruc):
                sset.add_structure(scell.gen_random(sub))
        else:
            raise Exception(' Sum of concentrations is not equal to 1')

        clx_structure = sset.get_structures()[0]
        self.alloyAtoms = clx_structure.get_atoms()  # ASE Atoms Class
        self.alloyStructure = AseAtomsAdaptor.get_structure(self.alloyAtoms)  # Pymatgen Structure
        self.alloyComposition = pmg.Composition(self.alloyAtoms.get_chemical_formula())  # Pymatgen Composition
        return self.alloyAtoms, lattice_param


    def gen_raw_crystal(self,crystalstructure, size,lattice_param = [None, None, None], element='X'):

        """
        generate a crystal filled with  element: 'element'
        """

        if crystalstructure== 'fcc':
            if all(elem  is None for elem in lattice_param):
                    raise Exception ('Please provide the lattice parameter')

            a= lattice_param[0]
            prime = bulk(name=element, crystalstructure='fcc', a=a, b=None, c=None)
            crystal = prime*tuple(size)
        elif crystalstructure== 'bcc':
            if all(elem  is None for elem in lattice_param):
                    raise Exception ('Please provide the lattice parameter')

            a= lattice_param[0]
            prime = bulk(name=element, crystalstructure='bcc', a=a, b=None, c=None)
            crystal = prime*tuple(size)
        elif crystalstructure== 'hpc':

            raise Exception('hpc is not yet implemented')
        else:
            raise Exception('Only fcc abd bcc are implemented')

        return AseAtomsAdaptor.get_structure(crystal)

    def gen_random_crystal_3D(self,crystalstructure, element, size):
        """
        use the Pyxtal package to gererate a ramdom 3D crystal

        """
        if crystalstructure== 'fcc':
            try:
                metallic_crystal = random_crystal(225 , element, size, 1.0, tm="metallic")
            except Exception as e:
                raise Exception(e)
        elif crystalstructure== 'bcc':
            try:
                metallic_crystal = random_crystal(216 ,  element, size, 1.0, tm="metallic")
            except Exception as e:
                raise Exception(e)
        return metallic_crystal.to_pymatgen()

    def get_max_diff_elements(self,element_pool, concentrations, nb_atm):
            """
                compute the maximun number of different element base on concentration
                and the total number of atoms
            """
            max_diff = {}
            for elm in element_pool:
                max_diff[elm] = round(nb_atm * concentrations[elm])

            if sum(max_diff.values()) == nb_atm:
                self.max_diff_element = max_diff
                return self.max_diff_element
            elif sum(max_diff.values()) < nb_atm:
                for ielem, elm in enumerate(element_pool):
                    while sum(max_diff.values()) != nb_atm :
                        max_diff[elm] = max_diff[elm] + 1

                self.max_diff_element = max_diff
                return self.max_diff_element



    def sites_neighbor_list(self, structure, cutoff):
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
        center_indices, points_indices, offset_vectors, distances = structure.get_neighbor_list(cutoff+0.5)
        all_neighbors_list = []

        #all_distance_list = []
        for i in range(structure.num_sites):
            site_neighbor = points_indices[np.where(center_indices==i)]
            neighbor_distance = distances[np.where(center_indices==i)]
            inds = neighbor_distance.argsort()
            sortedNeighbor = site_neighbor[inds]
            sortedDistance = neighbor_distance[inds]

            sortedNeighbor=np.insert(sortedNeighbor, 0, i)
            #sortedDistance=np.insert(sortedDistance, 0, 0)
            all_neighbors_list.append(sortedNeighbor)
            #all_distance_list.append(sortedDistance)

        return all_neighbors_list

    def site_neighbor_list(self, structure, cutoff,site_number):
        """
        structure :  pymatgen structure class

        cutoff : distance cutoff

        return list  neighbors of the site "site_number"
        """
        center_indices, points_indices, offset_vectors, distances  = structure.get_neighbor_list(cutoff+0.5)
        neighbors_list = []


        site_neighbor = points_indices[np.where(center_indices==site_number)]
        neighbor_distance= distances[np.where(center_indices==site_number)]

        inds = neighbor_distance.argsort()
        sortedNeighbor = site_neighbor[inds]
        sortedDistance = neighbor_distance[inds]

        sortedNeighbor=np.insert(sortedNeighbor, 0, site_number)







        return sortedNeighbor


    def site_neighbor_list2(self, structure, cutoff,site_number):
        """
        structure :  pymatgen structure class

        cutoff : distance cutoff

        return list  neighbors of the site "site_number"
        """

        sites = structure.sites
        sites_neighbours = structure.get_neighbors(sites[site_number], cutoff+0.5)
        neighbours_list =[sites[site_number].species_string]
        neighbours_list.extend([neighbour.species_string for neighbour in  sites_neighbours])

        neighbours_list = ['X' if elm =='X0+'  else elm for elm in neighbours_list]
        return neighbours_list


    def get_neighbor_in_offset_zero(self,structure, cutoff):

        """
        structure :  pymatgen structure class

        cutoff : distance cutoff

        return list of neighbour in offset [0, 0, 0]
        """
        center_indices, points_indices, offset_vectors, distances  = structure.get_neighbor_list(cutoff*3/4)

        offset_list = []
        #all_distance_list = []
        for i in range(structure.num_sites):
            site_neighbor = points_indices[np.where(center_indices==i)]
            offset = offset_vectors[np.where(center_indices==i)]
            inds = [ ielem  for  ielem, elem in enumerate(offset) if np.all(elem==0)]

            offset_list.append(site_neighbor[inds])


        return offset_list


    def get_neighbors_type(self,all_neighbors_list, alloyAtoms):
        """
        neighbors_list: list of list
        return  atomic numbers and symbols of the neighbours list
        considered site is  not include in the list
        """

        atomic_numbers = alloyAtoms.numbers
        #symbols = alloyAtoms.symbols
        symbols = alloyAtoms.get_chemical_symbols()
        numbers_vec = []
        symbols_vec = []


        for nb_list in all_neighbors_list:
            numbers_vec.append([atomic_numbers [i] for i in  nb_list[1:]]) # exclude the fisrt atom because it is the site considered
            symbols_vec.append([symbols[i] for i in  nb_list[1:]])
        return np.array(numbers_vec), np.array(symbols_vec)

    def get_neighbor_type(self, neighbors_list, alloyAtoms):

        """
        neighbors_list: list
        return  atomic numbers and symbols of the neighbours list
        """
        atomic_numbers = alloyAtoms.numbers
        symbols = alloyAtoms.get_chemical_symbols()

        numbers_vec = [atomic_numbers[i] for i in  neighbors_list[1:]] # exclude the fisrt atom because it is the site considered
        symbols_vec = [symbols[i] for i in  neighbors_list[1:]]
        return np.array(numbers_vec), np.array(symbols_vec)


    def get_neighbors_type2(self,all_neighbors_list, alloyAtoms):
        """
        neighbors_list: list of list
        return  atomic number and symbol of the neighbours list
        The considered site is  not include in the list
        """
        atomic_numbers = alloyAtoms.numbers

        symbols = alloyAtoms.get_chemical_symbols()
        numbers_vec = []
        symbols_vec = []

        for nb_list in all_neighbors_list:
            numbers_vec.append([atomic_numbers [i] for i in  nb_list])
            symbols_vec.append([symbols[i] for i in  nb_list])
        return numbers_vec, symbols_vec


    def get_neighbor_type2(self,neighbors_list, alloyAtoms):

        """
        neighbors_list: list
        return  atomic number and symbol of the neighbours list
        """
        atomic_numbers = alloyAtoms.numbers

        symbols = alloyAtoms.get_chemical_symbols()


        numbers_vec = [atomic_numbers[i] for i in   neighbors_list]
        symbols_vec = [symbols[i] for i in   neighbors_list]
        return  np.array(numbers_vec), np.array(symbols_vec)



    def count_occurence_to_dict(self,arr,element_pool ):
        """
        count occurence in numpy array and
        return a list dictinnary
        """
        my_list = []
        try:
            lenght = arr.shape[0]
        except:
             lenght = len(arr)
        for i in range( lenght):
            unique, counts = np.unique(arr[i], return_counts=True)
            my_dict = dict(zip(unique, counts))
            for elem in element_pool:
                if elem not in my_dict.keys():
                    my_dict[elem] = 0
            my_list.append(my_dict)
        return my_list


    def count_neighbors_by_shell(self, neighbors_list,alloyAtoms, element_pool):
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

        numbers_vec,  symbols_vec= self.get_neighbors_type(neighbors_list,alloyAtoms)

        shells=np.hsplit(symbols_vec, [NNeighbours]) # split into 12 and 6 for fcc
        #shellss=np.hsplit(shells[1], [6])





        shell1 =  self.count_occurence_to_dict(shells[0], element_pool) # first 12 column
        shell2 =  self.count_occurence_to_dict(shells[1], element_pool ) # lat 6 Column
        #shell3 =  count_occurence_to_dict(shellss[1], element_pool )


        return shell1, shell2


    def count_neighbors_in_offset(self,neighbors_list,alloyAtoms, element_pool):
        """
        return dictionary
        for each atom number of neighbor of each type in the offset 0
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

        we return list of dictionanry with number of atom in offset [0. 0. 0.]


        [{'Ag': 2, 'Pd': 1},  # first atom
         {'Ag': 2, 'Pd': 3},  # second atom
         {'Ag': 2, 'Pd': 3},
         {'Ag': 3, 'Pd': 2},
         {'Ag': 3, 'Pd': 2},
         {'Ag': 2, 'Pd': 3},
         {'Ag': 2, 'Pd': 3},
         {'Pd': 3, 'Ag': 0}]  # ....
        """

        numbers_vec,  symbols_vec= self.get_neighbors_type2(neighbors_list,alloyAtoms)


        offset0 =  self.count_occurence_to_dict( symbols_vec, element_pool)

        return offset0

    def get_symbols_indexes(self, alloyAtoms):
        """
        retrun dictionanry with indexes of each type of atom
        """
        chemical_symbols =  alloyAtoms.get_chemical_symbols()
        element_pool = list(set(chemical_symbols))

        symbols_indexes ={}

        for elem in element_pool:
            symbols_indexes[elem] =[i for i,x in enumerate(chemical_symbols) if x == elem]
        return symbols_indexes

    def _a_around_b(self,a, b, shell, alloyAtoms):
        """
        shell: list of dictionanry with the neighbour of each atom
        symbols_indexes : dictionary with the indexes of each element in the structure
        a: symbol of element a
        b: symbol of element b
        return list of number of atom a aound b
        """
        symbols_indexes = self.get_symbols_indexes(alloyAtoms)
        return [shell[i][a] for i in symbols_indexes[b] ]

    def get_CN_list(self,shell, alloyAtoms):
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
                    CN_list[combi]  = self._a_around_b(atm1, atm2, shell, alloyAtoms)
                except KeyError:
                    CN_list[combi] = [0]
                except Exception as e:
                    raise Exception(e)
        return CN_list



    def get_coordination_numbers(self, alloyAtoms,cutoff):
        """

        """


        alloyStructure = AseAtomsAdaptor.get_structure(alloyAtoms)
        # generate neighbor list in offset [ 0, 0, 0]
        all_neighbors_list = self.sites_neighbor_list(alloyStructure, cutoff)

        neighbor_in_offset = self.get_neighbor_in_offset_zero(alloyStructure,cutoff)

        shell1, shell2 =  self.count_neighbors_by_shell(all_neighbors_list,alloyAtoms, self.element_pool)


        CN1_list =self.get_CN_list(shell1, alloyAtoms) # Coordination number
        CN2_list = self.get_CN_list(shell2, alloyAtoms)



        return CN1_list, CN2_list




    def get_atom_properties(self,element_pool, oxidation_state=None):
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

                props = [elm.number]
                try:
                    props.append(oxidation_state[elm.name])
                except Exception as e:
                    props.append(0)
                try:
                    props.append(VEC[elm.name])  # valence

                except Exception as e:
                    raise Exception(e)
                try:
                    props.append(elm.X)  # electronegativity
                except Exception as e:
                    raise Exception(e)
                try:
                    props.append(elm.group)  #
                except Exception as e:
                    raise Exception(e)
                # props.append(elm.block)#
                try:
                    props.append(elm.row)  #
                except Exception as e:
                    raise Exception(e)
                try:
                    props.append(int(elm.is_metal))
                except Exception as e:
                    raise Exception(e)
                try:
                    props.append(int(elm.is_transition_metal))
                except Exception as e:
                    raise Exception(e)
                try:
                    props.append(int(elm.is_alkali))
                except Exception as e:
                    raise Exception(e)
                try:
                    props.append(int(elm.is_alkaline))
                except Exception as e:
                    raise Exception(e)
                try:
                #props.append(int(elm.is_chalcogen))
                #props.append(int(elm.is_halogen))
                    props.append(int(elm.is_metalloid))
                except Exception as e:
                    raise Exception(e)
                try:
                    props.append(round(elm.atomic_radius, 2))
                except Exception as e:
                    raise Exception(e)

                properties[elm.name] = props


            properties['X'] = [0]*len(properties[element_pool[0]])
            return properties



    def get_input_vectors(self, all_neighbors_list, properties):
        """
        all_neighbors_list: list of array with the list the neighbors  of each atom
        properties: dictionary with properties of each atom type
        """

        self.species = np.array(self.alloyStructure.species)
        input_vectors = []
        scaler = StandardScaler()
        for nb_list in all_neighbors_list:
            input_vector = []
            for elm in self.species[nb_list]:
                prop = properties[elm.name]
                input_vector.append(prop)
            input_vector = scaler.fit_transform(input_vector)
            input_vectors.append(input_vector)
        return np.array(input_vectors)

    def get_input_vector(self,neighbors_list, properties,alloyAtoms):
        """
        neighbors_list: list of array with the list the neighbors  an atom
        properties: dictionary with properties of each atom type
        """
        #numbers_vec,  symbols_vec = self.get_neighbor_type2(neighbors_list, alloyAtoms)


        scaler = StandardScaler()

        input_vector = [ properties[elm] for elm in neighbors_list]
        input_vector = scaler.fit_transform(input_vector)
        return np.array(input_vector)


    def gen_configuration(self,
            structureX,
            element_pool,
            properties,
            cutoff,
            model,device):
        """
        structureX: empty structure to be filled ( N sites)
        element_pool : list of dieffernet species
        max_diff_element: maximum nunber of different species
        model: NN model to train
        return generated Atoms configuration
        """
        config = copy.deepcopy(structureX)
        replace = True  # default  False
        elems_in = []


        Natoms = structureX.num_sites

        for i in range(Natoms):
            if i==0:
                atm = random.choices(element_pool)[0]
            else:
                idx = element_pool.index(atm)


                neighbors_list = self.site_neighbor_list2(config, cutoff, i)
                atomX = AseAtomsAdaptor.get_atoms(config)
                # numbers_vec, symbols_vec = AlloyGen.get_neighbor_type2(
                #    neighbors_list, atomX)

                input_vector = self.get_input_vector(
                    neighbors_list, properties, atomX)
                X_tensor = torch.from_numpy(input_vector).float()
                input_tensor = Variable(X_tensor, requires_grad=False).to(device)
                output_tensor = model(input_tensor)


                choice = output_tensor.multinomial(num_samples=1, replacement=replace)
                atm1 = element_pool[choice]

                # while atm1==atm:  # We have the max of this elements
                #    prob = output_tensor[idx]  # take the proba of the element
                #    prob = prob / (len(element_pool) - 1)  # divide by nber element -1
                #    output_tensor = output_tensor + prob  # add the value to each component
                #    output_tensor[idx] = 0  # set the selected proba to 0
                #    choice = output_tensor.multinomial(
                #        num_samples=1, replacement=replace)
                #    atm1 = element_pool[choice]
                atm = atm1

            replace_species = ReplaceSiteSpeciesTransformation(
                indices_species_map={i: atm})
            config = replace_species.apply_transformation(config)

            elems_in.append(atm)

        #init_weight = copy.deepcopy(model.l1.weight.data)
        atms = AseAtomsAdaptor.get_atoms(config)
        compo = pmg.Composition(atms.get_chemical_formula())
        fractional_composition = compo.fractional_composition

        return atms

    def gen_constrained_configuration(self,
                structureX,
                element_pool,
                properties,
                max_diff_element,
                cutoff,
                model,device):
            """
            add a contrain to the for the max_diff_element

            structureX: empty structure to be filled ( N sites)
            element_pool : list of dieffernet species
            max_diff_element: maximum nunber of different species
            model: NN model to train
            return generated Atoms configuration
            """
            config = copy.deepcopy(structureX)
            replace = True  # default  False
            elems_in = []
            max_diff_elem = copy.deepcopy(max_diff_element)


            Natoms = structureX.num_sites

            if sum(max_diff_element.values()) != Natoms:
                raise Exception('The number of site in the structre is not equal to sum of different elements')

            for i in range(Natoms):
                if i==0:
                    atm = random.choices(element_pool)[0]
                else:
                    idx = element_pool.index(atm)

                    neighbors_list = self.site_neighbor_list2(config, cutoff, i)
                    atomX = AseAtomsAdaptor.get_atoms(config)
                    # numbers_vec, symbols_vec = AlloyGen.get_neighbor_type2(
                    #    neighbors_list, atomX)

                    input_vector = self.get_input_vector(
                        neighbors_list, properties, atomX)
                    X_tensor = torch.from_numpy(input_vector).float()
                    input_tensor = Variable(X_tensor, requires_grad=False).to(device)
                    output_tensor = model(input_tensor)

                    choice = output_tensor.multinomial(num_samples=1, replacement=replace)
                    atm1 = element_pool[choice]



                    # while atm1==atm:  # We have the max of this elements
                    #    prob = output_tensor[idx]  # take the proba of the element
                    #    prob = prob / (len(element_pool) - 1)  # divide by nber element -1
                    #    output_tensor = output_tensor + prob  # add the value to each component
                    #    output_tensor[idx] = 0  # set the selected proba to 0
                    #    choice = output_tensor.multinomial(
                    #        num_samples=1, replacement=replace)
                    #    atm1 = element_pool[choice]
                    atm = atm1


                    idx = element_pool.index(atm)

                    while max_diff_elem[atm] == 0:  # We have the max of this elements
                        prob = output_tensor[idx]  # take the proba of the element
                        prob = prob / (len(element_pool) - 1)  # divide by nber element -1
                        output_tensor = output_tensor + prob  # add the value to each component
                        output_tensor[idx] = 0  # set the selected proba to 0
                        choice = output_tensor.multinomial(
                            num_samples=1, replacement=replace)
                        atm = element_pool[choice]
                        idx = element_pool.index(atm)

                max_diff_elem[atm] -= 1

                replace_species = ReplaceSiteSpeciesTransformation(
                    indices_species_map={i: atm})
                config = replace_species.apply_transformation(config)

                elems_in.append(atm)

            #init_weight = copy.deepcopy(model.l1.weight.data)
            atms = AseAtomsAdaptor.get_atoms(config)
            compo = pmg.Composition(atms.get_chemical_formula())
            fractional_composition = compo.fractional_composition

            return atms

