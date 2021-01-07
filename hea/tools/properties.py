import numpy as np
import yaml

from hea.tools import constants
from hea.tools import heatofmixing
from hea.tools import miedema
from hea.tools import valence_electron

PROPERTIES = [
    'atomic_size_difference',
    'mixing_entropy',
    'mixing_enthalpy',
    'VEC',
    'electronegativity',
    'melting_point',
    'omega',
    'phi',
    'miedema_energy',
]

__all__ = ['Property']


class Property:
    def __init__(self, structure):
        """
        Param structure: pymatgen class object
        with the information on the structure
        """
        self.structure = structure

    @staticmethod
    def get_property_names(name):
        """
        param:name:either usual name or operator name of a property
        return:( usual name, operator name + ending _deriv part) of the property
        """
        if name in PROPERTIES:
            usname = name
        else:
            usname = None
        return usname

    def get_property(self, name):
        """
        get any type of property
        param name:name of the property
        """
        # Get properties name and check it
        usname = self.get_property_names(name)

        if usname is None:  # invalid name
            print(f'Property [{name}] not recognized')
            return None

        try:
            prop = eval('self.get_' + usname + '()')
            return prop
        except AttributeError:
            print(f'Property [{usname}] not implemented')
            return None

    def get_atomic_size_difference(self):
        atom_list = self.structure.elements
        r_bar = 0.0
        for element in atom_list:
            r_bar += self.structure.get_atomic_fraction(element) * element.atomic_radius
        r_d = 0.0
        for element in atom_list:
            r_d += self.structure.get_atomic_fraction(element) * ((1 - (element.atomic_radius / r_bar)) ** 2)

        return 100 * np.sqrt(r_d)

    def get_mixing_entropy(self):
        atom_list = self.structure.elements
        S = 0.0
        for element in atom_list:
            S += self.structure.get_atomic_fraction(element) * np.log(self.structure.get_atomic_fraction(element))
        return -1 * S  # * Constants.R

    # def get_mixing_enthalpy(self):
    #    natoms = self.res.mol.natoms

    #   return delta_H

    def get_vec(self):
        atom_list = self.structure.elements  # [Element Fe, Element Ni]

        VEC = 0.0
        valence = yaml.safe_load(open('Tools/data/VEC.yml').read())

        for element in atom_list:
            # VEC += self.structure.get_atomic_fraction(element) * valence_electron.valence[str(element)]
            VEC += self.structure.get_atomic_fraction(element) * valence[str(element)]
        return VEC

    def get_electronegativity(self):
        atom_list = self.structure.elements

        X_bar = 0.0
        for element in atom_list:
            X_bar += self.structure.get_atomic_fraction(element) * element.X
        X_d = 0.0
        for element in atom_list:
            X_d += self.structure.get_atomic_fraction(element) * (element.X - X_bar) ** 2

        return np.sqrt(X_d)

    # def get_omega(self):
    #    natoms = self.res.mol.natoms

    #   return omega

    def get_melting_point(self):
        atom_list = self.structure.elements
        melting = 0.0
        for element in atom_list:
            melting += self.structure.get_atomic_fraction(element) * element.melting_point
        return melting

    def get_miedema_energy(self, composition=None):
        """

        :return: miedema energy
        """
        if composition is None:
            composition = self.structure
        energy = miedema.miedema.get(composition)
        return energy

    def get_mixing_enthalpy(self):
        """
        delta_H =  sum_ij 4(H_ij*c_i*c_j)
        H_ij : miedema  enthalpy of mixing of the binary liquid ij
                between the ith and jth elements at an equiatomic composition.
        :return: mixing energy
        """
        # atom_list = self.structure.elements
        atom_list = list(self.structure.as_dict().keys())

        mixing_enegy = 0
        for i in range(len(atom_list)):
            for j in range(i + 1, len(atom_list)):
                composition = atom_list[i] + atom_list[j]
                # miedema_energy = self.get_miedema_energy(composition)
                miedema_energy = heatofmixing.miedema.get(composition)
                c_i = self.structure.get_atomic_fraction(atom_list[i])
                c_j = self.structure.get_atomic_fraction(atom_list[j])

                mixing_enegy += miedema_energy * c_i * c_j

        return mixing_enegy * 4.0
