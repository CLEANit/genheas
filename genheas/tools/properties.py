import json
import os

import mendeleev
import numpy as np
import yaml

from genheas.utilities.log import logger
from pymatgen.core.periodic_table import Element
from sklearn.preprocessing import MinMaxScaler


__all__ = ["Property", "atomic_properties", "list_of_elements", "AtomInitializer", "AtomJSONInitializer"]
blocks = {"s": 1, "p": 2, "d": 3, "f": 4}
atomic_properties = {
    "atomic_number": "number",
    "group_number": "group",
    "period_number": "row",
    "block": "block",
    # "is an alkaline metal": 'is_metal',
    # "is_transition_metal": 'is_transition_metal',
    # "is_alkali": 'is_alkali',
    # "is_alkaline": 'is_alkaline',
    # "is_metalloid": 'is_metalloid',
    "first_ionization_energy": "ionenergies",
    "electron_affinity": "electron_affinity",
    "atomic_radius": "atomic_radius",
    "atomic_mass": "atomic_mass",
    "atomic_volume": "atomic_volume",
    # "is_chalcogen": 'is_chalcogen',
    # "oxidation_states": "oxidation_states",
    "Valence_electron_concentration": "VEC",
    "electronegativity": "X",
}

list_of_elements = [
    "Li",
    "Be",
    "Na",
    "Mg",
    "Al",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "H",
    "B",
    "N",
    "O",
    "C",
]

loc = os.path.dirname(os.path.abspath(__file__))
atom_init_file = os.path.join(loc, "data/atom_init.json")


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.
    !!! Use one AtomInitializer per dataset !!!
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}
        self.atomic_properties = atomic_properties
        self.list_of_elements = list_of_elements

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, "_decodedict"):
            self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}
        return self._decodedict[idx]

    def write_atom_init_json(self):
        atom_props = Property()
        atomproperties = {"X": [0.0] * len(self.atomic_properties)}
        for atom in sorted(self.atom_types):
            value = [atom_props.get_property(prop, atom) for prop in self.atomic_properties]
            atomproperties[atom] = value
        with open(atom_init_file, "w") as fp:
            json.dump(atomproperties, fp, indent=6)
        logger.info("writing atom_init.json")
        return atomproperties


class AtomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.
    Parameters
    ----------
    elem_embedding_file: str
        The path to the .json file
    """

    def __init__(self, elem_embedding_file=atom_init_file, rewrite=False):
        if os.path.exists(elem_embedding_file) and rewrite:
            os.remove(elem_embedding_file)

        if os.path.exists(elem_embedding_file):
            logger.info("reading atom_init.json")
            with open(elem_embedding_file) as f:
                elem_embedding = json.load(f)
            elem_embedding = {key: value for key, value in elem_embedding.items()}
            atom_types = set(elem_embedding.keys())
            super(AtomJSONInitializer, self).__init__(atom_types)
            for key, value in elem_embedding.items():
                self._embedding[key] = np.array(value, dtype=float)
        else:
            logger.info("computing atom_init file")
            super(AtomJSONInitializer, self).__init__(list_of_elements)
            atomfeatures = super(AtomJSONInitializer, self).write_atom_init_json()
            for key, value in atomfeatures.items():
                self._embedding[key] = np.array(value, dtype=float)


class Property:
    def __init__(self):
        """ """
        self.VEC_data = self._load_vec_data()
        self.atomic_properties = atomic_properties
        self.list_of_elements = list_of_elements
        self.property_data = self._get_property_data()

    @staticmethod
    def _load_vec_data():

        """
        Load VEC data from "data/VEC.yml"
        :return:
        """

        loc = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(loc, "data/VEC.yml")) as fr:
            data = yaml.safe_load(fr)
        return data

    def get_property_names(self, name):
        """
        param:name:either usual name or operator name of a property
        return:( usual name, operator name + ending _deriv part) of the property
        """
        if name in atomic_properties:
            usname = name
            opname = atomic_properties[name]
        # name is a operator name ?

        elif name in list(self.atomic_properties.values()):
            opname = name
            usname = list(atomic_properties.keys())[list(self.atomic_properties.values()).index(name)]
        else:
            opname = usname = None
        return usname, opname

    def get_property(self, name, elemen):
        """
        get any type of property
        param name:name of the property
        """
        # Get properties name and check it
        names = self.get_property_names(name)
        operator = names[1].split(",")[0]

        if name is None:  # invalid name
            logger.info(f"Property [{name}] not recognized")
            return None

        try:
            prop = eval('self.get_{}("{}")'.format(operator, elemen))
            return prop
        except AttributeError:
            logger.error(f"{AttributeError}")
            raise Exception(f'Property ["{name}"] not implemented')
            # return None

    def _get_property_data(self):
        """
        get any type of property
        param name:name of the property
        """
        property_data = {}

        # Get properties name and check it
        for name in self.atomic_properties:
            names = self.get_property_names(name)
            operator = names[1].split(",")[0]

            if name is None:  # invalid name
                logger.info(f"Property [{name}] not recognized")
                return None

            try:
                property_data[operator] = eval("self.get_{}_datas()".format(operator))

            except AttributeError:
                logger.error(f"{AttributeError}")
                raise Exception(f'Property ["{name}"] not implemented')
                # return None
        return property_data

    def _fitted_scaler(self, feature_name):
        """
        Compute the minimum and maximum to be used for later scaling.
        :param feature_name:
        :return: Fitted scaler.
        """
        # scaler.fit_transform(number.reshape(-1, 1)).flatten()
        scaler = MinMaxScaler(feature_range=(-1, 1))
        return scaler.fit(feature_name.reshape(-1, 1))

    def _transfromed_data(self, value, scaled_datas):
        """

        :param value: value to  transform (float)
        :param scaled_datas: array of data
        :return: Transformed value ( float value from array)
        """
        value = np.array([value])
        return scaled_datas.transform(value.reshape(-1, 1)).flatten()[0]

    def get_number(self, elm):
        """
        :param elm: str(atomic symbol)
        :return: transformed value between -1 and 1
        """
        val = Element(elm).number
        val = self._transfromed_data(val, self.property_data["number"])
        return round(val, 2)

    def get_number_datas(self):
        """
        :param elm: str(atomic symbol)
        :return: transformed value between -1 and 1
        """
        datas = np.array([Element(elmt).number for elmt in list_of_elements])
        return self._fitted_scaler(datas)

    def get_group(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        val = Element(elm).group
        val = self._transfromed_data(val, self.property_data["group"])
        return round(val, 2)

    def get_group_datas(self):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        datas = np.array([Element(elemt).group for elemt in list_of_elements])
        return self._fitted_scaler(datas)

    def get_block(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        val = blocks[Element(elm).block]
        val = self._transfromed_data(val, self.property_data["block"])
        return round(val, 2)

    def get_block_datas(self):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        datas = np.array([blocks[Element(elemt).block] for elemt in list_of_elements])
        return self._fitted_scaler(datas)

    def get_row(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        val = Element(elm).row
        val = self._transfromed_data(val, self.property_data["row"])
        return round(val, 2)

    def get_row_datas(self):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        datas = np.array([Element(elemt).row for elemt in list_of_elements])
        return self._fitted_scaler(datas)

    def get_atomic_radius(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        val = Element(elm).atomic_radius
        val = self._transfromed_data(val, self.property_data["atomic_radius"])
        return round(val, 2)

    def get_atomic_radius_datas(self):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        datas = np.array([Element(elemt).atomic_radius for elemt in list_of_elements])
        return self._fitted_scaler(datas)

    def get_atomic_mass(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        val = Element(elm).atomic_mass
        val = self._transfromed_data(val, self.property_data["atomic_mass"])
        return round(val, 2)

    def get_atomic_mass_datas(self):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        datas = np.array([Element(elemt).atomic_mass for elemt in list_of_elements])
        return self._fitted_scaler(datas)

    def get_atomic_volume(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        val = mendeleev.element(elm).atomic_volume
        if val is None:
            val = 0.0
        val = self._transfromed_data(val, self.property_data["atomic_volume"])
        return round(val, 2)

    def get_atomic_volume_datas(self):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        datas = np.array([mendeleev.element(elemt).atomic_volume for elemt in list_of_elements])
        datas = np.where(datas == None, 0.0, datas)
        return self._fitted_scaler(datas)

    def get_electron_affinity(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """

        val = mendeleev.element(elm).electron_affinity
        if val is None:
            val = 0.0
        val = self._transfromed_data(val, self.property_data["electron_affinity"])
        return round(val, 2)

    def get_electron_affinity_datas(self):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        datas = np.array([mendeleev.element(elemt).electron_affinity for elemt in list_of_elements])
        datas = np.where(datas == None, 0.0, datas)
        return self._fitted_scaler(datas)

    def get_ionenergies(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        val = mendeleev.element(elm).ionenergies[1]
        val = self._transfromed_data(val, self.property_data["ionenergies"])
        return round(val, 2)

    def get_ionenergies_datas(self):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        datas = np.array([mendeleev.element(elemt).ionenergies[1] for elemt in list_of_elements])
        return self._fitted_scaler(datas)

    def get_X(self, elm):
        """
        electronegativity
        :param elm: str(atomic symbol)
        :return:
        """

        val = Element(elm).X
        val = self._transfromed_data(val, self.property_data["X"])
        return round(val, 2)

    def get_X_datas(self):
        """
        electronegativity
        :param elm: str(atomic symbol)
        :return:
        """

        datas = np.array([Element(elemt).X for elemt in list_of_elements])
        return self._fitted_scaler(datas)

    def get_VEC(self, elm):
        """
        Valence electron concentration
        :param VEC: valence electron data
        :param elm: str(atomic symbol)
        :return:
        """
        val = self.VEC_data[elm]
        val = self._transfromed_data(val, self.property_data["VEC"])
        return round(float(val), 2)

    def get_VEC_datas(self):
        """
        Valence electron concentration
        :param VEC: valence electron data
        :param elm: str(atomic symbol)
        :return:
        """

        datas = np.array([self.VEC_data[element] for element in list_of_elements])
        return self._fitted_scaler(datas)

    # def get_is_metal(self, elm):
    #     """
    #     :param elm: str(atomic symbol)
    #     :return:
    #     """
    #     datas = np.array([float(Element(elemt).is_metal) for elemt in list_of_elements])
    #     val = Element(elm).is_metal
    #
    #     # val = self._transfromed_data(val, datas)
    #     return round(float(val), 2)
    #
    # def get_is_transition_metal(self, elm):
    #     """
    #     :param elm: str(atomic symbol)
    #     :return:
    #     """
    #
    #     datas = np.array([float(Element(elemt).is_transition_metal) for elemt in list_of_elements])
    #     val = Element(elm).is_transition_metal
    #
    #     # val = self._transfromed_data(val, datas)
    #     return round(float(val), 2)
    #
    # def get_is_alkali(self, elm):
    #     """
    #     :param elm: str(atomic symbol)
    #     :return:
    #     """
    #
    #     datas = np.array([float(Element(elemt).is_alkali) for elemt in list_of_elements])
    #     val = Element(elm).is_alkali
    #
    #     # val = self._transfromed_data(val, datas)
    #     return round(float(val), 2)
    #
    # def get_is_alkaline(self, elm):
    #     """
    #     :param elm: str(atomic symbol)
    #     :return:
    #     """
    #
    #     datas = np.array([float(Element(elemt).is_alkaline) for elemt in list_of_elements])
    #     val = Element(elm).is_alkaline
    #
    #     # val = self._transfromed_data(val, datas)
    #     return round(float(val), 2)
    #
    # def get_is_metalloid(self, elm):
    #     """
    #     :param elm: str(atomic symbol)
    #     :return:
    #     """
    #
    #     datas = np.array([float(Element(elemt).is_metalloid) for elemt in list_of_elements])
    #     val = Element(elm).is_metalloid
    #
    #     # val = self._transfromed_data(val, datas)
    #     return round(float(val), 2)
    #
    # def get_is_chalcogen(self, elm):
    #     """
    #     :param elm: str(atomic symbol)
    #     :return:
    #     """
    #
    #     datas = np.array([float(Element(elemt).is_chalcogen) for elemt in list_of_elements])
    #     val = Element(elm).is_metalloid
    #
    #     # val = self._transfromed_data(val, datas)
    #     return round(float(val), 2)
