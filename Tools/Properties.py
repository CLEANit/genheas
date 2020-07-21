


PROPERTIES = ["Atomic_size_difference", "missing_entropy", "mixing_enthalpy", "VEC",
              "electronegativity", "melting_point", "omega", "phi"]

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






class Property(object):

    def __init__(self, structure):
        """
        Param structure: pymatgen class object
        with the information on the structure
        """
        self.structure = structure




    def get_property(self, name,  atomlist=None):
        """
        get any type of property
        param name:name of the property
        Param atomlist:list of atoms for the mechanical properties
        """
        # Get properties name and check it
        usname = get_property_names(name)

        if usname is None: #invalid name
            print("Property [{}] not recognized".format(name))
            return None

        try:
            aproperty = eval("self.res.get_" + usname)
        except AttributeError:
            print("Property [{}] not implemented".format(usname))
            return None

        # check if the property was actually read
        if aproperty is None:
            print("Property [{}] not found/read correctly in file ['{}']".format(usname, self.res.filename))
            return None







    def get_atomic_size_difference(self):
        natoms = self.structure.num_atoms
        for i in len(natoms):

        return delta

    def get_mixing_entropy(self):
        natoms = self.res.mol.natoms

        return delta_S

    def get_mixing_enthalpy(self):
        natoms = self.res.mol.natoms

        return delta_H

    def get_VEC(self):
        natoms = self.res.mol.natoms

        return VEC

    def get_electronegativity(self):
        natoms = self.res.mol.natoms

        return delta_X

    def get_omega(self):
        natoms = self.res.mol.natoms

        return omega

    def get_melting_point(self):
        natoms = self.res.mol.natoms

        return Tm









class ListProperty(object):

    def __init__(self, files, w=None, keys=None):
        """
        Param files:list of file class
        Param w:angular frequencies of the perturbed field
        Param keys = list of keys to be associated with each property
              if None, filename is used
        """
        self.files = files
        self.lwl = w
        self.keys = keys
        if self.keys is None:
            self.keys = [file.filename for file in self.files]
        self.properties = {}
        for key, file in zip(self.keys, self.files):
            self.properties[key] = Property(file, w=self.lwl)

    def get_mechproperties(self, name, rotmatrix=None, atomlist=None):
        """
        get any type of mechanical property
        param name:name of the property
        Param rotmatrix:rotation matrix to apply on the properties with new axis as lines(U^dagger)
        Param atomlist:list of atoms for the mechanical properties
        """
        props = {}
        for key in self.properties:
            props[key] = self.properties[key].get_mechproperty(name, rotmatrix=rotmatrix, atomlist=atomlist)

        return props

    def get_properties(self, name, w=None, ws=False, rotmatrix=None, atomlist=None):
        """
        get any type of property
        param name:name of the property or name of the various operators from the response function
                    add _deriv_c for cartesian first-order derivative
                    add _deriv_c2 for cartesian second-order derivative
        param w:float or tuple with pulsations
        param ws:For quantity with pulsation, return the value for a specific pulsation(default) or a dictionary with all the pulsations
        Param rotmatrix:rotation matrix to apply on the properties with new axis as lines(U^dagger)
        Param atomlist:list of atoms for the mechanical properties
        """
        props = {}
        for key in self.properties:
            props[key] = self.properties[key].get_property(name, w=w, ws=ws, rotmatrix=rotmatrix, atomlist=atomlist)

        return props

