import numpy as np
from Tools import valence_electron
from Tools import Constants

PROPERTIES = ["atomic_size_difference", "mixing_entropy", "mixing_enthalpy", "VEC",
              "electronegativity", "melting_point", "omega", "phi"]


class Property(object):

    def __init__(self, structure):
        """
        Param structure: pymatgen class object
        with the information on the structure
        """
        self.structure = structure

    def get_property_names(self, name):
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
            print("Property [{}] not recognized".format(name))
            return None

        try:
            prop = eval("self.get_" + usname + "()")
            return prop
        except AttributeError:
            print("Property [{}] not implemented".format(usname))
            return None

    def get_atomic_size_difference(self):
        atom_list = self.structure.elements
        r_bar = 0.0
        for element in atom_list:
            r_bar += self.structure.get_atomic_fraction(element) * element.atomic_radius
        r_d = 0.0
        for element in atom_list:
            r_d += self.structure.get_atomic_fraction(element) * (1 - (element.atomic_radius / r_bar)) ** 2

        return np.sqrt(r_d)

    def get_mixing_entropy(self):
        atom_list = self.structure.elements
        S = 0.0
        for element in atom_list:
            S += self.structure.get_atomic_fraction(element) * np.log(self.structure.get_atomic_fraction(element))
        return -1 * S  # * Constants.R

    # def get_mixing_enthalpy(self):
    #    natoms = self.res.mol.natoms

    #   return delta_H

    def get_VEC(self):
        atom_list = self.structure.elements

        VEC = 0.0

        for element in atom_list:
            VEC += self.structure.get_atomic_fraction(element) * valence_electron.valence[str(element)]
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
