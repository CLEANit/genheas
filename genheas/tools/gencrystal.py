import copy
import itertools
import random
import sys

import numpy as np
import torch
import torch.nn.functional as f

from ase.build import bcc100
from ase.build import bcc110
from ase.build import bcc111
from ase.build import bulk
from ase.build import fcc100
from ase.build import fcc110
from ase.build import fcc111
from ase.build import hcp0001
from ase.build import hcp10m10
from ase.data import atomic_numbers
from ase.data import reference_states
from ase.lattice.cubic import BodyCenteredCubic
from ase.lattice.cubic import FaceCenteredCubic
from clusterx.parent_lattice import ParentLattice
from clusterx.structures_set import StructuresSet
from clusterx.super_cell import SuperCell
from genheas.tools.properties import AtomJSONInitializer
from genheas.utilities.log import logger
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.transformations.site_transformations import (
    ReplaceSiteSpeciesTransformation,
)
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
from tqdm import tqdm


# from ase.lattice.hexagonal import Hexagonal, HexagonalClosedPacked
# from pymatgen import Element


# https://en.wikipedia.org/wiki/Terrace_ledge_kink_model

coordination_numbers = {
    "fcc": [12, 6, 24],
    "bcc": [8, 6, 12],
    "hpc": [12, 6, 2],
    "fcc111": [9, 0, 0],
    "fcc100": [8, 0, 0],
    "fcc110": [7, 0, 0],
    "bcc111": [4, 0, 0],
    "bcc100": [4, 0, 0],
    "bcc110": [6, 0, 0],
    "hcp0001": [9, 0, 0],
    "hcp10m10": [10, 0, 0],
}


class AlloysGen(object):
    """"""

    atom_properties = {}
    atom_features = AtomJSONInitializer()

    def __init__(self, element_pool, concentrations, crystalstructure, radius=8.0):

        self.element_pool = element_pool
        self.concentrations = concentrations
        self.crystalstructure = crystalstructure
        self.peers = AlloysGen.get_peers(element_pool)
        self.radius = radius
        self.input_size = None
        self.NN1 = int(coordination_numbers[self.crystalstructure][0])
        self.NN2 = int(coordination_numbers[self.crystalstructure][1])
        self.max_num_nbr = self.NN1

    # @staticmethod
    # def get_cutoff(name, crystalstructure, lattice_param=None):
    #     """
    #     :param name: element name
    #     :param crystalstructure:
    #     :param lattice_param:
    #     :return:
    #     """
    #
    #     if lattice_param is None:
    #         try:
    #             Z = atomic_numbers[name]
    #             ref = reference_states[
    #                 Z
    #             ]  # {'symmetry': 'bcc', 'a': 4.23} or {'symmetry': 'hcp', 'c/a': 1.622, 'a': 2.51}
    #             xref = ref["symmetry"]
    #             a = ref["a"]
    #             if "c/a" in ref.keys():
    #                 c = ref["c/a"] * ref["a"]
    #             else:
    #                 c = None
    #         except KeyError:
    #             raise KeyError("Please provide the lattice parameter")
    #     else:
    #         a = lattice_param[0]
    #         c = lattice_param[1]
    #
    #     # if xref == 'fcc' or 'bcc':
    #     #     return
    #     # else:
    #     #     return a, c
    #     return c

    @staticmethod
    def get_peers(element_pool):
        """
        combinations list of  unique pair of atom in the structure
        """

        combinations = []
        for i in range(len(element_pool)):
            for j in range(len(element_pool)):
                combinations.append(element_pool[i] + "-" + element_pool[j])
        return combinations
        # pairs = []
        # for i, j in itertools.combinations_with_replacement(element_pool, 2):
        #     pairs.append(i + "-" + j)
        # return pairs

    @staticmethod
    def gen_alloy_supercell(
        element_pool,
        concentrations,
        crystalstructure,
        size,
        nb_structure,
        lattice_param=None,
        cubic=False,
    ):
        """
        :param nb_structure:
        :param cubic:
        :param lattice_param:
        :param element_pool: list of element in the alloy
        :param concentrations:
        :param crystalstructure: fcc or bcc
        :param size:
        :return: Alloy supercell
        """
        # if lattice_param is None:
        #     lattice_param = [None, None, None]

        prim = []
        nstruc = nb_structure

        if crystalstructure == "fcc":
            if lattice_param is None:
                try:
                    lattice_param = FaceCenteredCubic(element_pool[0]).cell.cellpar()[:3]
                except ValueError:
                    logger.error("You need to specify the lattice constant")
                    raise ValueError('No reference lattice parameter "a" for "{}"'.format(element_pool[0]))

            a = lattice_param[0]
            prime = bulk(name="X", crystalstructure="fcc", a=a, b=None, c=None, cubic=cubic)
            for elm in element_pool:
                prime_copy = copy.deepcopy(prime)
                prime_copy.set_chemical_symbols(elm)

                prim.append(prime_copy)

        elif crystalstructure == "bcc":
            if lattice_param is None:
                try:
                    """# array([ 3.52,  3.52,  # 3.52, 90.  , 90.  , 90.  ])"""
                    lattice_param = BodyCenteredCubic(element_pool[0]).cell.cellpar()[:3]

                except Exception as err:
                    logger.error(f"{err}")
                    raise Exception("Please provide the lattice parameter")

            a = lattice_param[0]
            prime = bulk(name="X", crystalstructure="bcc", a=a, b=None, c=None, cubic=cubic)

            for elm in element_pool:
                prime_copy = copy.deepcopy(prime)
                prime_copy.set_chemical_symbols(elm)

                prim.append(prime_copy)

        elif crystalstructure == "hpc":

            raise Exception("hpc is not yet implemented")
        else:
            raise Exception("Only fcc abd bcc are defined")

        platt = ParentLattice(prim[0], substitutions=prim[1:])
        scell = SuperCell(platt, size)

        # lattice_param = prim[0].cell[0][0]
        sset = StructuresSet(platt)
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
            raise Exception(" Sum of concentrations is not equal to 1")

        clx_structure = sset.get_structures()
        alloyAtoms = [structure.get_atoms() for structure in clx_structure]  # ASE Atoms Class

        return alloyAtoms, lattice_param

    @staticmethod
    def get_lattice_param(name):
        """
        :param name: Element symbol
        :return:
        """
        try:
            Z = atomic_numbers[name]
            ref = reference_states[Z]  # {'symmetry': 'bcc', 'a': 4.23} or {'symmetry': 'hcp', 'c/a': 1.622, 'a': 2.51}
            # xref = ref["symmetry"]
            a = ref["a"]
            if "c/a" in ref.keys():
                c = ref["c/a"] * ref["a"]
            else:
                c = None
        except KeyError:
            raise KeyError("Please provide the lattice parameter")
        return a, c

    @staticmethod
    def gen_crystal(
        crystalstructure,
        cel_size,
        max_diff_elem=None,
        lattice_param=None,
        name=None,
        cubik=False,
        radom=False,
    ):
        """
        :param radom:
        :param cel_size:
        :param crystalstructure:
        :param max_diff_elem:
        :param lattice_param:
        :param name:
        :param cubik:
        :return:
        """

        symmetries = list(coordination_numbers.keys())
        if crystalstructure not in symmetries:
            logger.error(f'crystal structure  "{crystalstructure}" is not implemented')
            raise Exception(f" [{crystalstructure}] is not implemented ")

        if lattice_param is not None:
            a = lattice_param[0]
            c = lattice_param[1]
        elif max_diff_elem is not None:
            name = max(max_diff_elem, key=max_diff_elem.get)
            a, c = AlloysGen.get_lattice_param(name)
        elif name is not None:
            a, c = AlloysGen.get_lattice_param(name)
        else:
            a = 4
            c = a * 1.622
        if crystalstructure == "fcc":
            atoms = bulk(name="X", crystalstructure="fcc", a=a, cubic=cubik) * tuple(cel_size)
        elif crystalstructure == "fcc111":
            atoms = fcc111("X", a=a, size=cel_size, vacuum=4 * a, orthogonal=cubik)
        elif crystalstructure == "fcc100":
            atoms = fcc100("X", a=a, size=cel_size, vacuum=4 * a, orthogonal=cubik)
        elif crystalstructure == "fcc110":
            atoms = fcc110("X", a=a, size=cel_size, vacuum=4 * a, orthogonal=cubik)
        # TODO slab 101

        elif crystalstructure == "bcc":
            atoms = bulk(name="X", crystalstructure="bcc", a=a, cubic=cubik) * tuple(cel_size)
        elif crystalstructure == "bcc111":
            atoms = bcc111("X", a=a, size=cel_size, vacuum=4 * a, orthogonal=cubik)
        elif crystalstructure == "bcc100":
            atoms = bcc100("X", a=a, size=cel_size, vacuum=4 * a, orthogonal=cubik)
        elif crystalstructure == "bcc110":
            atoms = bcc110("X", a=a, size=cel_size, vacuum=4 * a, orthogonal=cubik)
        # TODO slab 101

        elif crystalstructure == "hpc":
            atoms = bulk(name="X", crystalstructure="bcc", a=a, b=None, c=c, cubic=cubik) * tuple(cel_size)
        elif crystalstructure == "hpc0001":
            atoms = hcp0001("X", a=a, size=cel_size, vacuum=4 * a, orthogonal=cubik)
        elif crystalstructure == "hcp10m10":
            atoms = hcp10m10("X", a=a, size=cel_size, vacuum=4 * a, orthogonal=cubik)

        if radom and max_diff_elem is not None:
            elements_list = []
            try:
                for key, value in max_diff_elem.items():
                    elements_list.extend([key] * value)
                random.shuffle(elements_list)
                atoms.set_chemical_symbols(elements_list)
            except Exception as e:
                logger.error(f"{e}")
                raise Exception(f"{e}")

        elif name is not None:
            atoms.set_chemical_symbols([name] * len(atoms))

        return AseAtomsAdaptor.get_structure(atoms)

    def get_sites_neighbor_list(self, crystal, max_num_nbr=None, radius=None, site_number=None):

        if max_num_nbr is None:
            max_num_nbr = int(self.max_num_nbr)
        if radius is None:
            radius = self.radius
        all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x.nn_distance) for nbrs in all_nbrs]
        nbr_type_shell1, nbr_type_shell2 = [], []
        nbr_idx = []  # neighbors_index,
        nbr_type = []  # neighbors_symbol
        # distances = []  # neighbors_distances,
        for inbr, nbr in enumerate(all_nbrs):
            if len(nbr) < max_num_nbr:
                logger.warning("not find enough neighbors please consider increase ")
                nbr_idx.append([inbr] + list(map(lambda x: x.index, nbr)) + [0] * (max_num_nbr - len(nbr)))
                types = list(map(lambda x: x.specie.name, nbr)) + [0] * (max_num_nbr - len(nbr))
                nbr_type_shell1.append(types[: self.NN1])
                nbr_type_shell2.append(types[self.NN1:])
                nbr_type.append(types)
                # distances.append(
                #     list(map(lambda x: x.nn_distance, nbr)) + [radius + 1.0] * (max_num_nbr - len(nbr))
                # )
            else:
                nbr_idx.append([inbr] + list(map(lambda x: x.index, nbr[:max_num_nbr])))  # add
                types = list(map(lambda x: x.specie.name, nbr[:max_num_nbr]))
                nbr_type.append(types)  # add
                nbr_type_shell1.append(types[: int(self.NN1)])
                nbr_type_shell2.append(types[int(self.NN1):])
                # distances.append(list(map(lambda x: x.nn_distance, nbr[:max_num_nbr])))  # add dist 0

        nbr_type_shell1, nbr_type_shell2 = np.array(nbr_type_shell1), np.array(nbr_type_shell1)
        nbr_idx = np.array(nbr_idx)
        nbr_type = np.array(nbr_type)
        # distances = np.array(distances)

        if site_number is None:
            return nbr_idx, nbr_type, (nbr_type_shell1, nbr_type_shell2)
        elif 0 <= site_number <= len(all_nbrs):
            return (
                nbr_idx[site_number],
                nbr_type[site_number],
                (nbr_type_shell1[site_number], nbr_type_shell2, [site_number]),
            )
        else:
            raise Exception("site number out off index")

    def get_max_diff_elements(self, nb_atm):
        """
        compute the maximum number of different element base on concentration
        and the total number of atoms
        """

        logger.info("Coordination numbers Initialized ")
        max_diff = {}
        for elm in self.element_pool:
            max_diff[elm] = round(nb_atm * self.concentrations[elm])
        if sum(max_diff.values()) == nb_atm:
            return max_diff
        elif sum(max_diff.values()) < nb_atm:
            # add successively an atom  to each type of element
            for elm in self.element_pool:
                while sum(max_diff.values()) != nb_atm:
                    max_diff[elm] = max_diff[elm] + 1

        elif sum(max_diff.values()) > nb_atm:
            # Reversing a list
            # remove  successively an atom  to each type of element
            reversed_element_pool = self.element_pool[::-1]
            for elm in reversed_element_pool:
                while sum(max_diff.values()) != nb_atm:
                    max_diff[elm] = max_diff[elm] - 1

        # self.get_target_shell1()
        # self.get_target_shell2()
        self.max_diff_element = max_diff
        logger.info("max_diff_element Initialized ")

        return self.max_diff_element

    def get_input_vector(self, crystal, atm):
        """
        all_neighbors_list: list of array with the list the neighbors  of each atom
        properties: dictionary with properties of each atom type
        apply transformation by site
        :return atom_fea Tensor shape (nbr_neighbors, nbr_atom_fea )
        """
        # species = np.array(crystal.species)

        neighbor_list, neighbor_type, _ = self.get_sites_neighbor_list(crystal, site_number=atm)

        atom_fea = np.vstack([self.atom_features.get_atom_fea(crystal[i].specie.name) for i in neighbor_list])

        # row, col = atom_fea.shape
        # scaler1 = StandardScaler()
        # # scaler2 = MaxAbsScaler()
        # #
        # # # Standardized by column
        # atom_fea = scaler1.fit_transform(atom_fea.reshape(-1, atom_fea.shape[-1])).reshape(atom_fea.shape)
        # #
        # # atom_fea = scaler2.fit_transform(atom_fea)
        # atom_fea = atom_fea.reshape(-1, row, col)
        return torch.Tensor(atom_fea)

    def generate_configuration(
        self,
        config,
        element_pool,
        models,
        device,
        max_diff_element=None,
        constrained=False,
        verbose=False,
    ):
        """
        add a contrain to the for the max_diff_element

        structureX: empty structure to be filled ( N sites)
        element_pool : list of dieffernet species
        max_diff_element: maximum nunber of different species
        model: NN model to train
        return generated Atoms configuration
        """

        replace = True  # default  False
        elems_in = []

        Natoms = config.num_sites

        if max_diff_element is not None:
            if sum(max_diff_element.values()) != Natoms:
                raise Exception("The number of site in the structure is not equal to sum of different elements")
            else:
                max_diff_elem = copy.deepcopy(max_diff_element)

        for i in tqdm(range(Natoms), file=sys.stdout, leave=verbose):

            X_tensor = self.get_input_vector(config, i)

            with torch.no_grad():
                # X_tensor = torch.from_numpy(input_vector).float()
                input_tensor = Variable(X_tensor, requires_grad=False).to(device)

                output_tensors = [model(input_tensor) for model in models]
                output_tensor = torch.mean(torch.stack(output_tensors), dim=0)  # average
                output_tensor = f.normalize(output_tensor, p=1, dim=0)  # Normalization

            choice = torch.argmax(output_tensor)
            # choice = output_tensor.multinomial(num_samples=1, replacement=replace)

            atm = element_pool[choice]

            if constrained and max_diff_element is not None:
                idx = element_pool.index(atm)

                while max_diff_elem[atm] == 0 and len(elems_in) < Natoms:  # We have the max of this elements
                    atm, output_tensor = self._apply_constraint(output_tensor, element_pool, idx)
                    idx = element_pool.index(atm)

                max_diff_elem[atm] -= 1

            replace_species = ReplaceSiteSpeciesTransformation(indices_species_map={i: atm})
            config = replace_species.apply_transformation(config)

            elems_in.append(atm)
            # sys.stdout.flush()
        # atms = AseAtomsAdaptor.get_atoms(config)
        # atms.set_chemical_symbols(elems_in)

        return config
