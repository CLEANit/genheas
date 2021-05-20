#!/usr/bin/env python
"""
Created on Fri Oct 16 23:27:41 2020

@author: Conrard Tetsassi
"""
import datetime
import os
import pathlib
import pickle
import shutil
import time

import numpy as np
import torch
import yaml

from ase.io import write
from genheas.tools.evolution import NnEa
from genheas.tools.feedforward import Feedforward
from genheas.tools.gencrystal import AlloysGen
from genheas.tools.gencrystal import coordination_numbers
from genheas.tools.properties import atomic_properties
from genheas.utilities.log import logger
from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor


def read_model(path, element_pool, device, crystal_structure):
    nn_in_shell1 = coordination_numbers[crystal_structure][0]
    nn_in_shell2 = coordination_numbers[crystal_structure][1]
    n_neighbours = nn_in_shell1 + 1 + nn_in_shell2

    inputsize = n_neighbours * len(atomic_properties)

    outputsize = len(element_pool)

    my_model = Feedforward(inputsize, outputsize)
    my_model.to(device)

    # weights = read_policy()
    # model.l1.weight = torch.nn.parameter.Parameter(weights)

    try:
        my_model.load_state_dict(torch.load(path))
    except Exception as err:
        logger.error(f"{err}")
        raise Exception(f"{err}")
    return my_model


def read_policy(file):
    assert os.path.exists(file), f"{file} does not exist!"
    try:
        with open(file, "rb") as f:
            weights = pickle.load(f)
            return weights
    except pickle.UnpicklingError as e:
        # normal, somewhat expected
        logger.error(f"{e}")
    except (AttributeError, EOFError, ImportError, IndexError) as e:
        # secondary errors
        logger.error(f"{e}")
    except Exception as e:
        # everything else, possibly fatal
        logger.error(f"{e}")
        return


def atom_to_cif(name, atom):
    formula = atom.get_chemical_formula()
    write(f"{name}/{formula}.cif", atom)


def structure_to_cif(name, structure):
    formula = structure.composition.alphabetical_formula.replace(" ", "")
    structure.to(fmt="cif", filename=os.path.join(name, formula + ".cif"))


def write_to_png(name, configuration):
    formula = configuration.get_chemical_formula()
    write(f"{name}/{formula}.png", configuration)


def generate_structure(
    my_model,
    output_name,
    element_pool,
    conc,
    cryst_structure,
    cell_size,
    cell_param,
    constraints=True,
    cubik=False,
    device="cpu",
    verbose=False,
):
    logger.info("Start Structures Generation")
    since = time.time()

    pathlib.Path(output_name).mkdir(parents=True, exist_ok=True)

    GaNn = NnEa(element_pool, conc, cryst_structure)
    AlloyGen = AlloysGen(element_pool, conc, cryst_structure)

    nb_atom = len(
        AlloyGen.gen_crystal(
            cryst_structure,
            cell_size,
            max_diff_elem=None,
            lattice_param=None,
            name=None,
            cubik=cubik,
            radom=False,
        ),
    )

    max_diff_elements = AlloyGen.get_max_diff_elements(nb_atom)

    structureX = AlloyGen.gen_crystal(
        cryst_structure,
        cell_size,
        max_diff_elem=None,
        lattice_param=cell_param,
        name=element_pool[0],
        cubik=cubik,
        radom=False,
    )

    logger.info(f"Number of atoms : {nb_atom}")
    logger.info(f"Template structure : {structureX.composition.alphabetical_formula}")
    logger.info(f"Target max_diff_elements : {max_diff_elements}")

    configuration = AlloyGen.generate_configuration(
        structureX,
        element_pool,
        my_model,
        device=device,
        max_diff_element=max_diff_elements,
        constrained=constraints,
        verbose=verbose,
    )

    structure_to_cif(output_name, configuration)

    print("\n", "*" * 20, "Best structure", "*" * 20)

    print("Generated chemical formula: ", configuration.composition.alphabetical_formula, "\n")

    print("Fractional composition: ", configuration.composition.fractional_composition, "\n")

    target_shell1 = GaNn.get_target_shell1()
    target_shell2 = GaNn.get_target_shell2()

    _, _, shells = AlloyGen.get_sites_neighbor_list(configuration)

    shell1 = GaNn.count_occurrence_to_dict(shells[0], element_pool)
    shell2 = GaNn.count_occurrence_to_dict(shells[1], element_pool)

    CN1_list = GaNn.get_CN_list(shell1, configuration)  # Coordination number
    CN2_list = GaNn.get_CN_list(shell2, configuration)

    shell1_fitness_AA, shell1_fitness_AB = GaNn.get_mae_shell(CN1_list, target_shell1)
    shell2_fitness_AA, shell2_fitness_AB = GaNn.get_mae_shell(CN2_list, target_shell2)
    composition_fitness = GaNn.get_composition_fitness(max_diff_elements, configuration)

    fitness = (
        composition_fitness
        + shell1_fitness_AA
        # + shell1_fitness_AB
        # + shell2_fitness_AA
        # + shell2_fitness_AB
    )

    print("shell1_fitness_AA: {:6.2f}".format(shell1_fitness_AA))
    # print(f"shell1_fitness_AB: {shell1_fitness_AB:6.2f}")
    # print(f"shell2_fitness_AA: {shell2_fitness_AA:6.2f}")
    # print("shell2_fitness_AB: {:6.2f}".format(shell2_fitness_AB))
    print(f"composition_fitness: {composition_fitness:6.2f}\n")
    print(f"Total fitness: {fitness:6.2f}\n")

    print(f"<.cif> files have been saved in [{output_name}]")
    print("Total time for the generation h:m:s ::  {}".format(str(datetime.timedelta(seconds=time.time() - since))))
    print("*" * 60)
    ###########################################################################

    target_comp = Composition(max_diff_elements)
    script = "\n"
    script += f"Total fitness (shell2_AA +  max_diff_element): {fitness:6.2f}\n"
    script += "\n"
    script += f"shell1_fitness_AA: {shell1_fitness_AA:6.2f}\n"
    # script += "shell1_fitness_AB: {:6.2f}\n".format(shell1_fitness_AB))
    # script += "shell2_fitness_AA: {:6.2f}\n".format(shell2_fitness_AA))
    # script += f"shell2_fitness_AB: {shell2_fitness_AB:6.2f}\n"
    script += "\n"
    script += f"Target chemical formula: {target_comp}\n"
    script += f"Generated chemical formula: {configuration.composition}\n"
    script += f"max_diff_element_fitness: {composition_fitness:6.2f}\n"
    script += "\n"
    script += f"Target composition: {target_comp.fractional_composition}\n"
    script += f"Generated composition: {configuration.composition.fractional_composition}\n"
    script += f"composition_fitness: {composition_fitness:6.4f}\n"
    script += "\n"
    # f.write('shell1_fitness_AA: {:6.2f}\n'.format(sum(shell1_fitness_AA.values())))

    script += "Total time for the generation h:m:s ::  {}\n".format(
        str(datetime.timedelta(seconds=time.time() - since)),
    )

    script += "*" * 80
    script += "\n"
    formula = configuration.composition.alphabetical_formula.replace(" ", "")
    return script, formula, fitness


def main(policy_path=None):
    # ========================== Read Parameters  ============================

    input_file = "parameters.yml"
    try:
        with open(os.path.join("./", input_file)) as fr:
            params = yaml.safe_load(fr)
    except Exception as err:
        logger.error(f"{err}")
        raise Exception(f"{err}")

    crystal_structure = params["crystalstructure"]
    cell_parameters = params["cell_parameters"]
    elements_pool = params["elements_pool"]
    concentrations = params["concentrations"]
    device = params["device"]
    nb_structures = params["nb_structures"]
    supercell_size = params["supercell_size"]
    cube = params["cubic_gen"]

    # ==========================  Analyse Parameters  ========================

    # cutoff, cell_parameters = get_cutoff(cell_parameters, elements_pool, crystal_structure)
    if policy_path is None:
        policy_path = params["model_path"]

    NN_in_shell1 = coordination_numbers[crystal_structure][0]
    NN_in_shell2 = coordination_numbers[crystal_structure][1]
    NNeighbours = NN_in_shell1 + 1 + NN_in_shell2

    input_size = NNeighbours * len(atomic_properties)

    output_size = len(elements_pool)

    # ==========================  Load the model  ============================

    output = os.path.dirname(policy_path)

    best_policy = read_policy(policy_path)
    logger.info("Best ANNs weights have been read  from  pickle file")
    # os.remove(outfile)
    scripts = []
    fitnesses = []
    for i in range(nb_structures):
        networks = [Feedforward(input_size, output_size) for i in range(len(best_policy))]
        networks = [network.to(device) for network in networks]
        for j, w in enumerate(best_policy):
            networks[j].l1.weight = torch.nn.Parameter(w)
        scrip_t, formula, fitnes = generate_structure(
            networks,
            output,
            elements_pool,
            concentrations,
            crystal_structure,
            supercell_size,
            cell_parameters,
            constraints=False,
            cubik=cube,
            verbose=True,
        )

        shutil.move(os.path.join(output, formula + ".cif"), os.path.join(output, f"{i:06d}_{formula}.cif"))

        scripts.append(scrip_t)
        fitnesses.append(fitnes)

    fitnesses = np.array(fitnesses)
    indices = fitnesses.argsort()
    ofile = os.path.join(output, "generation.log")
    ofile = open(ofile, "w")
    ofile.write("# Generated structures sorted by total fitness")
    ofile.write("\n\n")
    for _, s in enumerate(indices):
        ofile.write(f'{"=" * 30} {s:06d} {"=" * 31}')
        ofile.write(scripts[s])
    ofile.close()


if __name__ == "__main__":
    main()
