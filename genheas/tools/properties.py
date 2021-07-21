import json
import os

from pathlib import Path

import mendeleev
import numpy as np
import torch
import yaml

from genheas.utilities.log import logger
from pymatgen.core.periodic_table import Element
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


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

atomic_properties_categories = {
    # "atomic_number": "number",
    "group": 18,
    "row": 9,
    "block": 4,
    # "first_ionization_energy": "ionenergies",
    # "electron_affinity": "electron_affinity",
    # "atomic_radius": "atomic_radius",
    # # "atomic_mass": "atomic_mass",
    # "atomic_volume": "atomic_volume",
    "VEC": 17,
    # "electronegativity": "X",
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

workdir = Path.cwd()
atom_init_file = os.path.join(loc, "data/atom_init.json")


# atom_init_file = os.path.join(workdir, "atom_init.json")


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""

        tensor = tensor.type(torch.float)
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


class AtomInitializer:
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
        atomproperties = {}
        for atom in sorted(self.atom_types):
            value = torch.hstack([atom_props.get_property(prop, atom) for prop in self.atomic_properties])
            atomproperties[atom] = value.tolist()

        key = list(atomproperties.keys())[0]
        atomproperties["X"] = [0.0] * len(atomproperties[key])
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
            super().__init__(atom_types)
            for key, value in elem_embedding.items():
                self._embedding[key] = np.array(value, dtype=float)
        else:
            logger.info("Initializing atom features ")
            super().__init__(list_of_elements)
            atomfeatures = super().write_atom_init_json()
            for key, value in atomfeatures.items():
                self._embedding[key] = np.array(value, dtype=float)


class Property:
    def __init__(self):
        """ """
        self.VEC_data = self._load_vec_data()
        self.atomic_properties = atomic_properties
        self.list_of_elements = list_of_elements
        self.normalizer = self._get_property_normalizer()
        # self.number_normilized = self.get_number_normalizer()
        # self.group_normilized = self.get
        # self.row_normilized = self.get
        # self.block_normilized = self.get
        # self.io_normilized = self.get
        # self.ea_normilized = self.get
        # self.radius_normilized = self.get
        # self.mass_normilized = self.get
        # self.volume_normilized = self.get
        # self.vec_normilized = self.get
        # self.X_normilized = self.get

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
            prop = eval(f'self.get_{operator}("{elemen}")')
            # logger.info(f"getting  {operator}")
            return prop
        except AttributeError:
            logger.error(f"{AttributeError}")
            raise Exception(f'Property ["{name}"] not implemented')
            # return None

    def _get_property_normalizer(self):
        """
        get any type of property
        param name:name of the property
        """
        normalizer = {}
        # Get properties name and check it
        for name in self.atomic_properties:
            names = self.get_property_names(name)
            operator = names[1].split(",")[0]

            if name is None:  # invalid name
                logger.info(f"Property [{name}] not recognized")
                return None

            try:
                logger.info(f"Normalizing { name}")
                normalizer[operator] = eval(f"self.get_{operator}_normalizer()")

            except AttributeError:
                logger.error(f"{AttributeError}")
                raise Exception(f'Property ["{name}"] not implemented')
                # return None
        return normalizer

    # def _fitted_scaler(self, feature_name):
    #     """
    #     Compute the minimum and maximum to be used for later scaling.
    #     :param feature_name:
    #     :return: Fitted scaler.
    #     """
    #     # scaler.fit_transform(number.reshape(-1, 1)).flatten()
    #     # scaler = MinMaxScaler(feature_range=(-1, 1))
    #     scaler = StandardScaler()
    #     # scaler = MaxAbsScaler()
    #     return scaler.fit(feature_name.reshape(-1, 1))

    # def _transfromed_data(self, value, scaled_datas):
    #     """
    #
    #     :param value: value to  transform (float)
    #     :param scaled_datas: array of data
    #     :return: Transformed value ( float value from array)
    #     """
    #     value = np.array([value])
    #     return scaled_datas.transform(value.reshape(-1, 1)).flatten()[0]

    def get_number(self, elm):
        """
        :param elm: str(atomic symbol)
        :return: transformed value between -1 and 1
        """
        if not isinstance(elm, list):
            elm = [elm]
        val = torch.tensor([Element(el).number for el in elm])
        val_normed = self.normalizer["number"].norm(val)
        # val_normed = self.number_normalizer.norm(val)

        return val_normed

    def get_number_normalizer(self):
        """
        :param elm: str(atomic symbol)
        :return: transformed value between -1 and 1
        """
        datas = torch.tensor([Element(elmt).number for elmt in list_of_elements])
        return Normalizer(datas)

    def get_group(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        if not isinstance(elm, list):
            elm = [elm]
        val = torch.tensor([Element(el).group for el in elm])
        val_normed = self.normalizer["number"].norm(val)
        # val_normed = self.group_normalizer.norm(val)

        return val_normed

    def get_group_normalizer(self):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        datas = torch.tensor([Element(elemt).group for elemt in list_of_elements])
        return Normalizer(datas)

    def get_block(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        if not isinstance(elm, list):
            elm = [elm]
        val = torch.tensor([blocks[Element(el).block] for el in elm])
        val_normed = self.normalizer["block"].norm(val)
        # val_normed = self.block_normalizer.norm(val)

        return val_normed

    def get_block_normalizer(self):
        """
        :param elm: str(atomic symbol)
        :return:
        """

        datas = torch.tensor([blocks[Element(elemt).block] for elemt in list_of_elements])
        return Normalizer(datas)

    def get_row(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        if not isinstance(elm, list):
            elm = [elm]
        val = torch.tensor([Element(el).row for el in elm])
        val_normed = self.normalizer["row"].norm(val)
        # val_normed = self.row_normalizer.norm(val)
        return val_normed

    def get_row_normalizer(self):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        datas = torch.tensor([Element(elemt).row for elemt in list_of_elements])
        return Normalizer(datas)

    def get_VEC(self, elm):
        """
        Valence electron concentration
        :param VEC: valence electron data
        :param elm: str(atomic symbol)
        :return:
        """
        if not isinstance(elm, list):
            elm = [elm]
        val = torch.tensor([self.VEC_data[el] for el in elm])

        # val_normed = self.vec_normalizer.norm(val)
        val_normed = self.normalizer["VEC"].norm(val)
        return val_normed

    def get_VEC_normalizer(self):
        """
        Valence electron concentration
        :param VEC: valence electron data
        :param elm: str(atomic symbol)
        :return:
        """

        datas = torch.tensor([self.VEC_data[element] for element in list_of_elements])

        return Normalizer(datas)

    def get_atomic_radius(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        if not isinstance(elm, list):
            elm = [elm]
        val = torch.tensor([Element(el).atomic_radius for el in elm])
        val_normed = self.normalizer["atomic_radius"].norm(val)
        # val_normed = self.radius.norm(val)

        return val_normed

    def get_atomic_radius_normalizer(self):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        datas = torch.tensor([Element(elemt).atomic_radius for elemt in list_of_elements])

        return Normalizer(datas)

    def get_atomic_mass(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        if not isinstance(elm, list):
            elm = [elm]
        val = torch.tensor([Element(el).atomic_mass for el in elm])
        val_normed = self.normalizer["atomic_mass"].norm(val)
        # val_normed = self.mass_normalizer.norm(val)

        return val_normed

    def get_atomic_mass_normalizer(self):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        datas = torch.tensor([Element(elemt).atomic_mass for elemt in list_of_elements])
        datas = torch.nan_to_num(datas)
        return Normalizer(datas)

    def get_atomic_volume(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        if not isinstance(elm, list):
            elm = [elm]
        val = [mendeleev.element(el).atomic_volume for el in elm]
        val = [i if i is not None else 0.0 for i in val]
        val = torch.tensor(val)

        val_normed = self.normalizer["atomic_volume"].norm(val)
        # val_normed = self.volume_normalizer.norm(val)

        return val_normed

    def get_atomic_volume_normalizer(self):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        datas = [mendeleev.element(elemt).atomic_volume for elemt in list_of_elements]
        datas = [i if i is not None else 0.0 for i in datas]
        datas = torch.tensor(datas)
        return Normalizer(datas)

    def get_electron_affinity(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        if not isinstance(elm, list):
            elm = [elm]
        val = [mendeleev.element(el).electron_affinity for el in elm]
        val = [i if i is not None else 0.0 for i in val]
        val = torch.tensor(val)
        val_normed = self.normalizer["electron_affinity"].norm(val)
        # val_normed = self.nea_normalizer.norm(val)

        return val_normed

    def get_electron_affinity_normalizer(self):
        """
        :param elm: str(atomic symbol)
        :return:
        """

        datas = [mendeleev.element(elemt).electron_affinity for elemt in list_of_elements]
        datas = [i if i is not None else 0.0 for i in datas]
        datas = torch.tensor(datas)
        return Normalizer(datas)

    def get_ionenergies(self, elm):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        if not isinstance(elm, list):
            elm = [elm]
        val = [mendeleev.element(el).ionenergies[1] for el in elm]
        val = [i if i is not None else 0.0 for i in val]
        val = torch.tensor(val)
        val_normed = self.normalizer["ionenergies"].norm(val)
        # val_normed = self.io_normalizer.norm(val)

        return val_normed

    def get_ionenergies_normalizer(self):
        """
        :param elm: str(atomic symbol)
        :return:
        """
        datas = [mendeleev.element(elemt).ionenergies[1] for elemt in list_of_elements]
        datas = [i if i is not None else 0.0 for i in datas]
        datas = torch.tensor(datas)
        return Normalizer(datas)

    def get_X(self, elm):
        """
        electronegativity
        :param elm: str(atomic symbol)
        :return:
        """
        if not isinstance(elm, list):
            elm = [elm]
        val = torch.tensor([Element(el).X for el in elm])
        val = torch.nan_to_num(val)
        val_normed = self.normalizer["X"].norm(val)
        # val_normed = self.X_normalizer.norm(val)
        return val_normed

    def get_X_normalizer(self):
        """
        electronegativity
        :param elm: str(atomic symbol)
        :return:
        """

        datas = torch.tensor([Element(elemt).X for elemt in list_of_elements])
        datas = torch.nan_to_num(datas)
        return Normalizer(datas)

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
