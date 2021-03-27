import os
import yaml

import numpy as np

from pymatgen import Element

from genheas.utilities.log import logger
from sklearn.preprocessing import MinMaxScaler

__all__ = ['Property', 'atomic_properties', 'list_of_elements']

# atomic_properties = [
#     'number',
#     'group',
#     'row',
#     # 'is_metal',
#     'is_transition_metal',
#     'is_alkali',
#     'is_alkaline',
#     'is_metalloid',
#     'atomic_radius',
#     # 'oxidation_states',
#     'VEC',
#     'electronegativity',
# ]

atomic_properties = {
    "atomic_number": 'number',
    "periodic_table_group": 'group',
    "periodic_table_row": 'row',
    # "is an alkaline metal": 'is_metal',
    "is_transition_metal": 'is_transition_metal',
    "is_alkali": 'is_alkali',
    "is_alkaline": 'is_alkaline',
    "is_metalloid": 'is_metalloid',
    "atomic_radius": 'atomic_radius',
    # "is_chalcogen": 'is_chalcogen',
    # "oxidation_states": "oxidation_states",
    "Valence_electron_concentration": 'VEC',
    "electronegativity": "X"
}

list_of_elements = ['Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V',
                    'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr',
                    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                    'In', 'Sn', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm',
                    'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
                    'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
                    'Bi', 'Po', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu',
                    'Am']


class Property:
    def __init__(self):
        """
        """
        self.VEC_data = self._load_vec_data()

    @staticmethod
    def _load_vec_data():

        """
        Load VEC data from "data/VEC.yml"
        :return:
        """

        loc = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(loc, 'data/VEC.yml')) as fr:
            data = yaml.safe_load(fr)
        return data

    @staticmethod
    def get_property_names(name):
        """
        param:name:either usual name or operator name of a property
        return:( usual name, operator name + ending _deriv part) of the property
        """
        if name in atomic_properties:
            usname = name
            opname = atomic_properties[name]
        # name is a operator name ?

        elif name in list(atomic_properties.values()):
            opname = name
            usname = list(atomic_properties.keys())[list(atomic_properties.values()).index(name)]
        else:
            opname = usname = None
        return usname, opname

    # def get_property_size(self):
    #
    #     return None

    def get_property(self, name, elemen):
        """
        get any type of property
        param name:name of the property
        """
        # Get properties name and check it
        names = self.get_property_names(name)
        operator = names[1].split(",")[0]

        if name is None:  # invalid name
            logger.info(f'Property [{name}] not recognized')
            return None

        try:
            prop = eval('self.get_{}("{}")'.format(operator, elemen))
            return prop
        except AttributeError:
            logger.error(f'{AttributeError}')
            raise Exception(f'Property ["{name}"] not implemented')
            # return None

    def _fitted_scaler(self, feature_name):
        """
        Compute the minimum and maximum to be used for later scaling.
        :param feature_name:
        :return: Fitted scaler.
        """
        # scaler.fit_transform(number.reshape(-1, 1)).flatten()
        scaler = MinMaxScaler(feature_range=(-1, 1))
        return scaler.fit(feature_name.reshape(-1, 1))

    def _transfromed_data(self, data, feature):
        """

        :param data: value to  transform (float)
        :param feature: name of the feature
        :return: Transformed data ( float value from array)
        """
        data = np.array([data])
        feature_fitted = self._fitted_scaler(feature)
        return feature_fitted.transform(data.reshape(-1, 1)).flatten()[0]

    # def fitter(self):
    #     """
    #     fit MinMaxScaler to each atomic_properties
    #     :return: dictionary with Fitted scaler object
    #     """
    #     mydict = {}
    #     for prop in atomic_properties:
    #         data = np.array([VEC_data[elm] for elm in list_of_elements])
    #         data = np.array([Element(elm).group / len(list_of_elements) for elm in list_of_elements])
    #
    #         mydict[prop] = self._fitted_scaler(data)
    #     return mydict

    def get_number(self, elm):
        """
        :param elm: str(atomic symbol)
        :return: transformed value between -1 and 1
        """
        datas = np.array([Element(element).number for element in list_of_elements])
        val = Element(elm).number
        return self._transfromed_data(val, datas)

    def get_group(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        datas = np.array([Element(element).group for element in list_of_elements])
        val = Element(elm).group
        return self._transfromed_data(val, datas)

    def get_row(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """

        datas = np.array([Element(element).row for element in list_of_elements])
        val = Element(elm).row
        return self._transfromed_data(val, datas)

    def get_atomic_radius(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        datas = np.array([Element(element).atomic_radius for element in list_of_elements])
        val = Element(elm).atomic_radius
        return self._transfromed_data(val, datas)

    def get_X(self, elm):
        """
        electronegativity
        :param elm: str(atomic symbol)
        :return:
        """

        datas = np.array([Element(element).X for element in list_of_elements])
        val = Element(elm).X
        return self._transfromed_data(val, datas)

    def get_VEC(self, elm):
        """
        Valence electron concentration
        :param VEC: valence electron data
        :param elm: str(atomic symbol)
        :return:
        """

        datas = np.array([self.VEC_data[element] for element in list_of_elements])
        val = self.VEC_data[elm]
        val = self._transfromed_data(val, datas)
        return val

    def get_is_metal(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        datas = np.array([float(Element(element).is_metal) for element in list_of_elements])
        val = Element(elm).is_metal

        val = self._transfromed_data(val, datas)
        return val

    def get_is_transition_metal(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """

        datas = np.array([float(Element(element).is_transition_metal) for element in list_of_elements])
        val = Element(elm).is_transition_metal

        val = self._transfromed_data(val, datas)
        return val

    def get_is_alkali(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """

        datas = np.array([float(Element(element).is_alkali) for element in list_of_elements])
        val = Element(elm).is_alkali

        val = self._transfromed_data(val, datas)
        return val

    def get_is_alkaline(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """

        datas = np.array([float(Element(element).is_alkaline) for element in list_of_elements])
        val = Element(elm).is_alkaline

        val = self._transfromed_data(val, datas)
        return val

    def get_is_metalloid(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """

        datas = np.array([float(Element(element).is_metalloid) for element in list_of_elements])
        val = Element(elm).is_metalloid

        val = self._transfromed_data(val, datas)
        return val
