#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 23:27:41 2020

@author: Conrard Tetsassi
"""

import numpy as np
from ase.lattice.cubic import FaceCenteredCubic, BodyCenteredCubic
from ase.build import bulk
import pymatgen as pmg
from pymatgen.io.ase import AseAtomsAdaptor

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structures_set import StructuresSet
from pymatgen import Element

import yaml

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.utils import to_categorical

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
import copy


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
        VEC = yaml.safe_load(open("Tools/data/VEC.yml").read())
        #species_set = list(set(species))
        species_oxidation_state = alloy_composition.oxi_state_guesses()

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


class Feedforward(nn.Module):
    def __init__(self, input_size, output_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.output_size, bias=True)
        # init.normal_(self.l1.weight, mean=0, std=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0),
                   -1)  # convert tensor (nb_atom, 1, nb_neighbors, nb_prop) --> (nv_atom,nb_neighbors*nb_prop)
        output = self.l1(x)
        output = self.softmax(output)
        return output


def gen_policies(Feedforward, input_size, output_size,  input_tensor, nb_policies):
        policies = []  # list of output vectors
        policies_weights = []
      
        
        for i in range(nb_policies):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Feedforward(input_size, output_size)
            model.to(device)
            input_tensor = input_tensor.to(device)
            output_tensor = model(input_tensor)
            output_vector = output_tensor.cpu().detach().numpy()
            policies.append(output_vector)

            init_weight = copy.deepcopy(model.l1.weight.data)
            init_weight = init_weight.cpu().detach().numpy()
            policies_weights.append(init_weight)
        return policies, policies_weights


def gen_structure(alloyatoms, out_vector, atoms_list_set):
    atms = copy.deepcopy(alloyatoms)
    for i in range(len(out_vector)):
        atm = np.random.choice(atoms_list_set, p=out_vector[i])
        atms.symbols[i] = atm
    atomic_fraction = {}
    compo = pmg.Composition( atms.get_chemical_formula())
    for elm in atoms_list_set:
        atomic_fraction[elm] = compo.get_atomic_fraction(Element(elm))
    return atms, atomic_fraction



def objective_function(fraction, target):
    """
    fractions is a dictionary with concentration of each atom
    target is a dictionary with the target concentration
    return the RMSD
    """
    summ = 0
    for key in fraction.keys():
        summ += (abs(fraction[key] -target[key] ))**2
        
    rmsd = np.sqrt(summ/len(fraction.keys()))  
    return rmsd


def get_policies_objective(alloyatoms, policies, atoms_list_set,target):

    policies_obj_funct= []
    for policy in policies:
        structures = []
        fractions = []
        obj_funct = []
        n =10
        for i in range(n):
            struct, frac = gen_structure(alloy_atoms,policy, atoms_list_set)
            fractions.append(frac)
            structures.append(struct)
        
        for comp in fractions:
            obj_funct.append(objective_function(comp, target))   
        policies_obj_funct.append(np.average(np.array(obj_funct))) 
    return policies_obj_funct     



if __name__ == "__main__":
    element_list = ['Ag', 'Pd']
    concentrations = {'Ag': 0.5, 'Pd': 0.5}
    target_concentration =  {'Ag': 0.5, 'Pd': 0.5}
    cell_type = 'fcc'
    cell_size = [4, 4, 4]

    AlloyGen = AlloysGen(element_list, concentrations, cell_type, cell_size)
    # generate alloys supercell
    alloy_atoms, lattice_param = AlloyGen.gen_alloy_supercell(elements=element_list,
                                                              concentrations=concentrations, types=cell_type,
                                                              size=cell_size)

    alloy_structure = AseAtomsAdaptor.get_structure(alloy_atoms)  # Pymatgen Structure
    alloy_composition = pmg.Composition(alloy_atoms.get_chemical_formula())  # Pymatgen Composition

    # generate neighbors list
    all_neighbors_list = AlloyGen.sites_neighbor_list(alloy_structure, lattice_param)

    # get properties of each atoms
    properties = AlloyGen.get_atom_properties(alloy_structure)

    # generate input vectors
    input_vectors = AlloyGen.get_input_vectors(all_neighbors_list, properties)

    # generate input tensor
    X_tensor = torch.from_numpy(input_vectors).float()
    input_tensor = Variable(X_tensor)  # requires_grad=False

    # Feedforward
    input_size = np.prod(input_vectors[0].shape)  # prod(18,14)
    output_size = output_size = len(set(alloy_atoms.get_chemical_symbols()))
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = MLP(input_size, output_size)
    #model.to(device)

    #input_tensor = input_tensor.to(device)

    # generate policies ang get objective function
    nb_policies = 10
    policies, policies_weights = gen_policies(Feedforward, input_size, output_size,  input_tensor, nb_policies)
    policies_obj_funct = get_policies_objective(alloy_atoms, policies, element_list,  target_concentration )
    
    size = int(np.ceil(len(policies)*0.25))
    top_index =  sorted(range(len(policies_obj_funct)), key=lambda i: policies_obj_funct[i])[-size:]

    top_policies = [policies[idx] for idx in top_index]
    top_policies_weights  =[ policies_weights[idx]for idx in top_index]
    
    

