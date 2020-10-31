import yaml
import numpy as np
from ase.lattice.cubic import FaceCenteredCubic, BodyCenteredCubic
# from ase.build import bulk
import pymatgen as pmg
from pymatgen.io.ase import AseAtomsAdaptor

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structures_set import StructuresSet

# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.utils import to_categorical

class AlloysGen(object):

    def __init__(self, element_list, concentration, cell_type, cell_size):
        self.element_list = element_list
        self.concentration = concentration
        self.cell_type = cell_type
        self.cell_size = cell_size
        self.species = None
        self.alloy_atoms = None
        self.alloy_structure = None
        self.alloy_composition = None

    def gen_alloy_supercell(self, elements, concentrations, types, size):
        """
        :param elements: list of element in the alloy
        :param concentrations:
        :param types: fcc or bcc
        :param size:
        :return: Alloy supercell
        """
        prim = []
        if types == 'fcc':
            for elm in elements:
                prim.append(FaceCenteredCubic(elm))
        else:
            for elm in elements:
                prim.append(BodyCenteredCubic(elm))

        platt = ParentLattice(prim[0], substitutions=prim[1:])
        scell = SuperCell(platt, size)
        lattice_param = prim[0].cell[0][0]
        sset = StructuresSet(platt)
        nstruc = 1
        nb_atm = []
        sub = {}
        for elm in elements:
            nb_atm.append(round(len(scell) * concentrations[elm]))
        if sum(nb_atm) == len(scell):
            sub[0] = nb_atm[1:]
            for i in range(nstruc):
                sset.add_structure(scell.gen_random(sub))
        else:
            raise Exception(' Sum of concentrations is not equal to 1')

        clx_structure = sset.get_structures()[0]
        self.alloy_atoms = clx_structure.get_atoms()  # ASE Atoms Class
        self.alloy_structure = AseAtomsAdaptor.get_structure(self.alloy_atoms)  # Pymatgen Structure
        self.alloy_composition = pmg.Composition(self.alloy_atoms.get_chemical_formula())  # Pymatgen Composition
        return self.alloy_atoms, lattice_param

    def sites_neighbor_list(self, structure, cutoff):
        """
        structure :  pymatgen structure class

        cutoff : distance cutoff

        return list of numpy array with the neighbors of each site
        """
        center_indices, points_indices, offset_vectors, distances = structure.get_neighbor_list(cutoff)
        all_neighbors_list = []
        for i in range(structure.num_sites):
            site_neighbor = points_indices[np.where(center_indices == i)]
            all_neighbors_list.append(site_neighbor)

        return all_neighbors_list

    def get_atom_properties(self, structure):
        """

        structure : pymatgen structure class
        return : dictionary with properties of each atom
        """
        species = np.array(structure.species)
        VEC = yaml.safe_load(open("tools/data/VEC.yml").read())

        composition = pmg.Composition(structure.formula)
        #composition = self.alloy_composition
        species_oxidation_state = composition.oxi_state_guesses()

        properties = {}

        for elm in set(species):
            props = [elm.number]
            if len(species_oxidation_state) == 0:  # oxidation_state
                props.append(0)
            else:
                props.append(species_oxidation_state[elm.name])  # oxidation_state
            props.append(VEC[elm.name])  # valence
            props.append(elm.X)  # electronegativity
            props.append(elm.group)  #
            # props.append(elm.block)#
            props.append(elm.row)  #
            props.append(int(elm.is_metal))
            props.append(int(elm.is_transition_metal))
            props.append(int(elm.is_alkali))
            props.append(int(elm.is_alkaline))
            props.append(int(elm.is_chalcogen))
            props.append(int(elm.is_halogen))
            props.append(int(elm.is_metalloid))
            props.append(round(elm.atomic_radius, 2))

            properties[elm.name] = props
        return properties

    def get_input_vectors(self, all_neighbors_list, properties):
        """
        all_neighbors_list: list of array with the list the neighbors  of each atom
        properties: dictionary with properties of each atom type
        """

        self.species = np.array(self.alloy_structure.species)
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

