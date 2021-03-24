import os
import yaml

import numpy as np

from pymatgen import Element

from genheas.utilities.log import logger

__all__ = ['Property', 'atomic_properties', 'list_of_elements']

atomic_properties = [
    'number',
    'group',
    'row',
    # 'is_metal',
    'is_transition_metal',
    'is_alkali',
    'is_alkaline',
    'is_metalloid',
    'atomic_radius',
    # 'oxidation_states',
    'VEC',
    'electronegativity',
]

list_of_elements = ['Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V',
                    'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr',
                    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                    'In', 'Sn', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm',
                    'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
                    'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
                    'Bi', 'Po', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu',
                    'Am']

loc = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(loc, 'data/VEC.yml')) as fr:
    VEC = yaml.safe_load(fr)


def transform_number():
    number = np.array([Element(elm).number / len(list_of_elements) for elm in list_of_elements])
    # number = (number - number.mean()) / number.std()
    return {'mean': number.mean(), 'std': number.std(), 'min': number.min(), 'max': number.max(),
            'maxabs': np.abs(number).max()}


def transform_group():
    group = np.array([Element(elm).group / len(list_of_elements) for elm in list_of_elements])
    # group = (group - group.mean()) / group.std()
    return {'mean': group.mean(), 'std': group.std(), 'min': group.min(), 'max': group.max(),
            'maxabs': np.abs(group).max()}


def transform_row():
    row = np.array([Element(elm).row / len(list_of_elements) for elm in list_of_elements])
    # row = (row - row.mean()) / row.std()
    return {'mean': row.mean(), 'std': row.std(), 'min': row.min(), 'max': row.max(), 'maxabs': np.abs(row).max()}


def transform_atomic_radius():
    atomic_radius = np.array([Element(elm).atomic_radius / len(list_of_elements) for elm in list_of_elements])
    # atomic_radius = (atomic_radius - atomic_radius.mean()) / atomic_radius.std()
    return {'mean': atomic_radius.mean(), 'std': atomic_radius.std(), 'min': atomic_radius.min(),
            'max': atomic_radius.max(), 'maxabs': np.abs(atomic_radius).max()}


def transform_electronegativity():
    electronegativity = np.array([Element(elm).X / len(list_of_elements) for elm in list_of_elements])
    # electronegativity = (electronegativity -  electronegativity.mean()) /  electronegativity.std()
    return {'mean': electronegativity.mean(), 'std': electronegativity.std(), 'min': electronegativity.min(),
            'max': electronegativity.max(), 'maxabs': np.abs(electronegativity).max()}


def transform_VEC(VEC=VEC):
    valence_elec = np.array([VEC[elm] / len(list_of_elements) for elm in list_of_elements])
    # valence_elec = (valence_elec - valence_elec.mean()) /valence_elec.std()
    return {'mean': valence_elec.mean(), 'std': valence_elec.std(), 'min': valence_elec.min(),
            'max': valence_elec.max(), 'maxabs': np.abs(valence_elec).max()}


class Property:
    def __init__(self):
        """
        """
        self.numbers = transform_number()
        self.groups = transform_group()
        self.rows = transform_row()
        self.atomic_radiis = transform_atomic_radius()
        self.electro_negativ = transform_electronegativity()
        self.valence_e = transform_VEC()
        self.nb_elements = len(list_of_elements)

    @staticmethod
    def get_property_names(name):
        """
        param:name:either usual name or operator name of a property
        return:( usual name, operator name + ending _deriv part) of the property
        """
        if name in atomic_properties:
            usname = name
        else:
            usname = None
        return usname

    def get_property(self, name, elemen):
        """
        get any type of property
        param name:name of the property
        """
        # Get properties name and check it
        usname = self.get_property_names(name)

        if usname is None:  # invalid name
            logger.info(f'Property [{name}] not recognized')
            return None

        try:
            prop = eval('self.get_{}("{}")'.format(usname, elemen))
            return prop
        except AttributeError:
            logger.error(f'{AttributeError}')
            raise Exception(f'Property ["{usname}"] not implemented')
            # return None

    def get_number(self, elm):
        """
        :param elm: str(atomic symbol)
        :return: transformed value between -1 and 1
        """
        number = Element(elm).number
        number = number / self.nb_elements
        # number = (number - self.numbers['mean']) / self.numbers['std']
        number = number / self.numbers['maxabs']
        return number

    def get_group(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        group = Element(elm).group
        group = group / self.nb_elements
        # group = (group - self.groups['mean']) / self.groups['std']
        group = group / self.groups['maxabs']
        return group

    def get_row(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        row = Element(elm).row
        row = row / self.nb_elements
        # row = (row - self.rows['mean']) / self.rows['std']
        row = row / self.rows['maxabs']

        return row

    def get_atomic_radius(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        atomic_radius = Element(elm).atomic_radius
        atomic_radius = atomic_radius / self.nb_elements
        # atomic_radius = (atomic_radius - self.atomic_radiis['mean']) / self.atomic_radiis['std']
        atomic_radius = atomic_radius / self.atomic_radiis['maxabs']
        return atomic_radius

    def get_electronegativity(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        X = Element(elm).X
        X = X / self.nb_elements
        # X = (X - self.electro_negativ['mean']) / self.electro_negativ['std']
        X = X / self.electro_negativ['maxabs']
        return X

    def get_VEC(self, elm, VEC=VEC):
        """
        :param VEC: valence electron data
        :param elm: str(atomic symbol)
        :return:
        """
        valence_elec = VEC[elm]
        valence_elec = valence_elec / self.nb_elements
        # valence_elec = (valence_elec - self.valence_e['mean']) / self.valence_e['std']
        valence_elec = valence_elec / self.valence_e['maxabs']
        return valence_elec

    @staticmethod
    def get_is_transition_metal(elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        return float(Element(elm).is_transition_metal)

    @staticmethod
    def get_is_transition_metal(elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        return float(Element(elm).is_transition_metal)

    @staticmethod
    def get_is_alkali(elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        return float(Element(elm).is_alkali)

    @staticmethod
    def get_is_alkaline(elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        return float(Element(elm).is_alkaline)

    @staticmethod
    def get_is_metalloid(elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        return float(Element(elm).is_metalloid)

    @staticmethod
    def get_is_metal(elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        return float(Element(elm).is_metal)
