import sys

import numpy as np
import pymatgen as pmg
import yaml

__all__ = ['Miedema']
# data rows:
# Element_name Phi Rho Vol Z Valence TM? RtoP Htrans
# Data format: | Element | Phi | Nws1/3 | Vm 2/3 | Z | Valence |Transition Metal (1) or Not (0)| R_cf| Htrans


# params = yaml.safe_load(open(qmpy.INSTALL_PATH + "/data/miedema.yml").read())
params = yaml.safe_load(open('Tools/data/miedema.yml').read())


class Miedema:
    def __init__(self, composition):
        """
        Takes a variety of composition representations and returns the miedema
        model energy, if possible.
        Examples::
            >>> get_miedema_energy({"Fe":0.5, "Ni":0.5})
            -0.03
            >>> get_miedema_energy({'Fe':2, 'Ni':2})
            -0.03
            >>> get_miedema_energy('FeNi')
            -0.03
            >>> composition = Composition.get('FeNi')
            >>> get_miedema_energy(composition)
            -0.03
        Returns:
            Energy per atom. (eV/atom)

        """
        self.mixing_energy = None
        # validate composition
        if isinstance(composition, str):
            composition = pmg.Composition(composition)
        elif isinstance(composition, pmg.Composition):
            pass
        elif isinstance(composition, dict):
            composition = pmg.Composition(composition)
        else:
            raise TypeError('Unrecognized composition:', composition)

        if len(composition) != 2:
            return None

        composition = composition.fractional_composition.as_dict()
        if not all(params[k] for k in list(composition.keys())):
            self.mixing_energy = None
            return

        # composition = composition.fractional_composition.as_dict()

        self.elt_a, self.elt_b = list(composition.keys())
        self.x = composition[self.elt_a]
        self.A = params[self.elt_a]
        self.B = params[self.elt_b]
        #
        self.A_phiStar = 0.0
        self.A_nws13 = 0.0
        self.A_Vm23 = 0.0
        self.aA = 0.0
        self.A_name = self.elt_a
        #
        self.B_phiStar = 0.0
        self.B_nws13 = 0.0
        self.B_Vm23 = 0.0
        self.aB = 0.0
        self.B_name = self.elt_b
        #
        self.deH_A_partial_infDilute = 0.0
        self.P = 0.0
        self.RP = 0.0
        self.QP = 9.4
        self.e = 1.0
        #
        self.elementName = {}
        self.elementPhiStar = {}
        self.elementNWS13 = {}
        self.elementVM23 = {}
        self.elementRP = {}
        self.elementTRAN = {}
        self.Avogardro = 6.02e23  # unit /mole
        #
        self.xA = np.linspace(0.001, 0.999, 200)
        self.xAs = np.empty(len(self.xA))
        self.fxs = np.empty(len(self.xA))
        self.g = np.empty(len(self.xA))
        self.deHmix = np.empty(len(self.xA))
        self.xB = 1.0 - self.xA
        self.__get_data()
        # self.mixing_energy = 0

        # for letter in ['A', 'B']:
        #  # Data format: | Element | Phi | Nws1/3 | Vm 2/3 | Z | Valence |Transition Metal (1) or Not (0)| R/P| Htrans
        #     element_data = eval("self."+letter)
        #     name = element_data[0]
        #     self.elementName[name] = element_data[0]
        #     self.elementPhiStar[name] = element_data[1]
        #     self.elementNWS13[name] = element_data[2]
        #     self.elementVM23[name] = element_data[3]
        #     self.elementRP[name] = element_data[7]
        #     self.elementTRAN[name] = element_data[6]

    def __get_data(self):
        """
        Initilaze parameter for elements A and B
        :return:
        """
        elements = {'A': self.A_name, 'B': self.B_name}
        for key in elements.keys():
            #  Element: Phi | Nws1/3 | Vm 2/3 | Z | Valence |Transition Metal (1) or Not (0)| R/P| Htrans
            element_data = eval('self.' + key)
            name = elements[key]
            self.elementName[name] = name
            self.elementPhiStar[name] = element_data[0]
            self.elementNWS13[name] = element_data[1]
            self.elementVM23[name] = element_data[2]
            self.elementRP[name] = element_data[6]
            self.elementTRAN[name] = element_data[5]
        return
        # @property

    def calRP(self):
        """
        Calculate and return the value of RtoP based on the transition metal
        status of elements A and B, and the elemental values of RtoP for elements A
        and B.
        """

        if self.elementTRAN[self.A_name] == 1 and self.elementTRAN[self.B_name] == 1:
            self.RP = 0.0
        elif self.elementTRAN[self.A_name] == 0 and self.elementTRAN[self.B_name] == 0:
            self.RP = 0.0
        else:
            self.RP = float(self.elementRP[self.A_name]) * float(self.elementRP[self.B_name]) * 0.73
        return

    def assiginP(self):
        """
        Chooses a value of P based on the transition metal status of the elements A and B.
        There are 3 values of P for the cases where:
        both A and B are TM
        only one of A and B is a TM
        neither are TMs.
        """
        if (self.elementTRAN[self.A_name] + self.elementTRAN[self.B_name]) == 2:
            self.P = 0.147
        elif (self.elementTRAN[self.A_name] + self.elementTRAN[self.B_name]) == 0:
            self.P = 0.111
        else:
            self.P = 0.128
        return

    def decideA(self):
        Alkali = ['Li', 'Na', 'K', 'Rb', 'Sc', 'Fr']
        for itm in Alkali:
            if self.elementName[self.A_name] == itm:
                self.aA = 0.14
            if self.elementName[self.B_name] == itm:
                self.aB = 0.14
        return

    def a_A(self):
        return self.pick_a(self.elt_a)

    # @property
    def a_B(self):
        return self.pick_a(self.elt_b)

    def pick_a(self, elt):
        """Choose a value of a based on the valence of element A."""
        possible_a = [0.14, 0.1, 0.07, 0.04]
        if elt == self.elt_a:
            params = self.A
        else:
            params = self.B
        if params[4] == 1:
            return possible_a[0]
        elif params[4] == 2:
            return possible_a[1]
        elif params[4] == 3:
            return possible_a[2]
        # elif elementA in ["Ag","Au","Ir","Os","Pd","Pt","Rh","Ru"]:
        elif elt in ['Ag', 'Au', 'Cu']:
            return possible_a[2]
        else:
            return possible_a[3]

    def calHmix(self):

        self.calRP()
        self.assiginP()
        # self.decideA()
        self.aA = self.a_A()
        self.aB = self.a_B()
        self.A_phiStar = float(self.elementPhiStar[self.A_name])
        self.B_phiStar = float(self.elementPhiStar[self.B_name])
        self.A_nws13 = float(self.elementNWS13[self.A_name])
        self.B_nws13 = float(self.elementNWS13[self.B_name])
        self.A_Vm23 = float(self.elementVM23[self.A_name])
        self.B_Vm23 = float(self.elementVM23[self.B_name])
        dePhi = self.A_phiStar - self.B_phiStar
        deNws13 = self.A_nws13 - self.B_nws13
        for i in range(len(self.xA)):
            self.A_Vm23Alloy = self.A_Vm23 * (1 + self.aA * self.xB[i] * (dePhi))
            self.B_Vm23Alloy = self.B_Vm23 * (1 + self.aB * self.xA[i] * (-1 * dePhi))
            self.xAs[i] = (
                self.xA[i] * self.A_Vm23Alloy / (self.xA[i] * self.A_Vm23Alloy + self.xB[i] * self.B_Vm23Alloy)
            )
            self.fxs[i] = self.xAs[i] * (1.0 - self.xAs[i])
            self.g[i] = (
                2.0
                * (self.xA[i] * self.A_Vm23Alloy + self.xB[i] * self.B_Vm23Alloy)
                / (1.0 / self.A_nws13 + 1.0 / self.B_nws13)
            )
            self.deHmix[i] = (
                self.Avogardro
                * self.fxs[i]
                * self.g[i]
                * self.P
                * (-self.e * (dePhi) ** 2 + self.QP * (deNws13) ** 2 - self.RP)
                * 1.60217657e-22
            )
        self.deH_A_partial_infDilute = (
            2.0
            * self.A_Vm23
            / (1.0 / self.A_nws13 + 1.0 / self.B_nws13)
            * self.Avogardro
            * self.P
            * (-self.e * (dePhi) ** 2 + self.QP * (deNws13) ** 2 - self.RP)
            * 1.60217657e-22
        )

        return

    def get_Hmix(self):

        self.calRP()
        self.assiginP()
        # self.decideA()
        self.aA = self.a_A()
        self.aB = self.a_B()
        #
        self.xA = self.x  # composition[self.elt_a]
        self.xAs = 0.0
        self.fxs = 0.0
        self.g = 0.0
        self.deHmix = 0.0
        self.xB = 1.0 - self.xA
        self.A_phiStar = float(self.elementPhiStar[self.A_name])
        self.B_phiStar = float(self.elementPhiStar[self.B_name])
        self.A_nws13 = float(self.elementNWS13[self.A_name])
        self.B_nws13 = float(self.elementNWS13[self.B_name])
        self.A_Vm23 = float(self.elementVM23[self.A_name])
        self.B_Vm23 = float(self.elementVM23[self.B_name])
        dePhi = self.A_phiStar - self.B_phiStar
        deNws13 = self.A_nws13 - self.B_nws13

        self.A_Vm23Alloy = self.A_Vm23 * (1 + self.aA * self.xB * dePhi)
        self.B_Vm23Alloy = self.B_Vm23 * (1 + self.aB * self.xA * (-1 * dePhi))
        self.xAs = self.xA * self.A_Vm23Alloy / (self.xA * self.A_Vm23Alloy + self.xB * self.B_Vm23Alloy)
        self.fxs = self.xAs * (1.0 - self.xAs)
        self.g = (
            2.0 * (self.xA * self.A_Vm23Alloy + self.xB * self.B_Vm23Alloy) / (1.0 / self.A_nws13 + 1.0 / self.B_nws13)
        )
        self.deHmix = (
            self.Avogardro
            * self.fxs
            * self.g
            * self.P
            * (-self.e * dePhi ** 2 + self.QP * deNws13 ** 2 - self.RP)
            * 1.60217657e-22
        )

        # self.report()
        return self.deHmix

    def report(self):
        print('')
        print('-------------------------------------------------')
        print('-------------------- report ---------------------')
        print(
            'Two components: ',
            self.A_name,
            '(',
            self.elementTRAN[self.A_name],
            ') and ',
            self.B_name,
            '(',
            self.elementTRAN[self.B_name],
            ')',
        )
        # print ('Conc of', self.A_name, self.x)
        # print ('Conc of', self.B_name, (1-self.x))
        print('Phi of', self.A_name, self.A_phiStar)
        print('Phi of', self.B_name, self.B_phiStar)
        print('n(ws)^1/3 of', self.A_name, self.A_nws13)
        print('n(ws)^1/3 of', self.B_name, self.B_nws13)
        print('Vm^2/3 of', self.A_name, self.A_Vm23)
        print('Vm^2/3 of', self.B_name, self.B_Vm23)
        print('a_A', self.aA)
        print('a_B', self.aB)
        print('P:', self.P)
        print('R/P:', self.RP)
        print('Q0/P:', self.QP)
        print('-------------------------------------------------')
        return

    @staticmethod
    def get(composition):
        # H = hmix()
        # calRP()
        # assiginP()
        # decideA()
        mixing_energy = Miedema(composition).get_Hmix()
        return mixing_energy
